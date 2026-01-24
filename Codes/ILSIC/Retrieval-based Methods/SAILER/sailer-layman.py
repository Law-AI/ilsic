import os
import sys
import json
import pickle as pkl
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from copy import deepcopy
from itertools import product
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer, AutoModel, BertModel
from transformers import BatchEncoding, TrainingArguments, Trainer, AdamW
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.file_utils import ModelOutput
from transformers.data.data_collator import DataCollatorMixin
from accelerate import Accelerator

TRAIN_DEV = True
TEST = True
SUFFIX = "layman-full"

NUM_CANDS = 48
EPOCHS = 5
BATCH_SIZE = 4
GRAD_ACC = 2

if not os.path.exists(f"/home/shounak/HDD/Layman-LSI/model/sailer-{SUFFIX}"):
    os.makedirs(f"/home/shounak/HDD/Layman-LSI/model/sailer-{SUFFIX}")

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# torch.manual_seed(2204459113277793439)
# torch.cuda.manual_seed_all(2204459113277793439)

with open("/home/shounak/HDD/Layman-LSI/dataset/layman-full-meta.json") as fr:
    metadata = json.load(fr)
stat2idx = {s:i for i,s in enumerate(metadata['stats'])}
qry2idx = {q:i for i,q in enumerate(metadata['test'])}    
# qry2idx = {q:i for i,q in enumerate(metadata['train'] + metadata['dev'])}

with open("/home/shounak/HDD/Layman-LSI/dataset/layman-full-gold.json") as fr:
    citations = json.load(fr)
    
with open("/home/shounak/HDD/Layman-LSI/model/sailer-layman-full-inference/negatives.json") as fr:
    neg_citations = json.load(fr)

with open("/home/shounak/HDD/Layman-LSI/dataset/section-texts.json") as fr:
    stat_textmap = json.load(fr)

with open("/home/shounak/HDD/Layman-LSI/dataset/layman-full-query-texts.json") as fr:
    qry_textmap = json.load(fr)


class SAILERDataset(Dataset):
    def __init__(self, ids, textmap, is_query=False, is_test=False, citations=None, neg_citations=None, num_neg_samples=64):
        self.dataset = []
        self.is_query = is_query
        self.is_test = is_test
        for id in tqdm(ids, desc="Building..."):
            data = {'id': id, 'text': textmap[id] if id in textmap else "[MASK]"}
            if self.is_query and not self.is_test:
                rel_stats = citations[id]
                data['stat_all_relevant'] = rel_stats
                for s in rel_stats:
                    data2 = deepcopy(data)
                    data2['stat_relevant'] = s
                    data2['stat_bm25_negatives'] = random.sample(neg_citations[id][:num_neg_samples], NUM_CANDS - 1)
                    self.dataset.append(data2)
            elif self.is_test:
                data['stat_relevant'] = citations[id]
                self.dataset.append(data)
            else:
                self.dataset.append(data)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def tokenize(self, tokenizer):
        for d in tqdm(self.dataset, desc='Tokenizing...'):
            d['input_ids'] = tokenizer(d['text'], add_special_tokens=False, return_tensors=None).input_ids
            
    def return_textmap(self):
        textmap = {}
        for d in self.dataset:
            textmap[d['id']] = d['input_ids']
        return textmap

tokenizer = AutoTokenizer.from_pretrained("CSHaitao/sailer_en_finetune")

print("TOKENIZING...")
if not os.path.exists(f"/home/shounak/HDD/Layman-LSI/model/sailer-{SUFFIX}/tokenized.pkl"):
    stat_dataset = SAILERDataset(metadata['stats'], stat_textmap)
    stat_dataset.tokenize(tokenizer)
    
    train_dataset = SAILERDataset(metadata['train'], qry_textmap, is_query=True, citations=citations, neg_citations=neg_citations)
    train_dataset.tokenize(tokenizer)
    
    dev_dataset = SAILERDataset(metadata['dev'], qry_textmap, is_query=True, citations=citations, neg_citations=neg_citations)#
    dev_dataset.tokenize(tokenizer)
    
    test_dataset = SAILERDataset(metadata['test'], qry_textmap, is_query=True, is_test=True, citations=citations, neg_citations=neg_citations)
    test_dataset.tokenize(tokenizer)
    
    with open(f"/home/shounak/HDD/Layman-LSI/model/sailer-{SUFFIX}/tokenized.pkl", 'wb') as fw:
        pkl.dump([stat_dataset, train_dataset, dev_dataset, test_dataset], fw)
    
else:
    with open(f"/home/shounak/HDD/Layman-LSI/model/sailer-{SUFFIX}/tokenized.pkl", 'rb') as fr:
        stat_dataset, train_dataset, dev_dataset, test_dataset = pkl.load(fr)
    
print(len(stat_dataset), len(train_dataset), len(dev_dataset), len(test_dataset))
 
stat_tok_textmap = stat_dataset.return_textmap()

# sys.exit(0)

@dataclass
class SAILERDataCollator(DataCollatorMixin):
    def __init__(self, stat_textmap, tokenizer, loss_type='categorical', cands_per_qry=8, chunk_size=512, overlap_size=128):
        self.stat_textmap = stat_textmap
        self.tokenizer = tokenizer
        self.cands_per_qry = cands_per_qry
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.loss_type = loss_type
  
    def divide_into_chunks(self, text):
        i = 0
        chunks = []
        while i < len(text):
            _text = text[i:i+self.chunk_size-2]
            chunk = [self.tokenizer.cls_token_id] + _text + [self.tokenizer.sep_token_id]
            chunk += [self.tokenizer.pad_token_id] * (self.chunk_size - len(chunk))
            chunks.append(chunk)
            i += self.chunk_size - 2 - self.overlap_size
        return chunks
           
    def __call__(self, queries):
        batch = {}
        for k in ['query_pos', 'query_input_ids', 'stat_input_ids']:
            # if 'input' not in k: continue
            batch[k] = []
        
        cand2idx = {}
        cand2idx['stat'] = {}
        cand2idx['prec'] = {}
        cand2idx['qry'] = {} 
        
        for qry in queries:
            # spos, ppos = [], []
            if qry['id'] not in cand2idx['qry']:
                cand2idx['qry'][qry['id']] = len(cand2idx['qry'])
                batch['query_input_ids'].append(torch.tensor(self.divide_into_chunks(qry['input_ids']), dtype=torch.long))
            batch['query_pos'].append(cand2idx['qry'][qry['id']])
            for s in [qry['stat_relevant']] + qry['stat_bm25_negatives']:
                if s not in cand2idx['stat']: cand2idx['stat'][s] = len(cand2idx['stat'])
            
        for s in cand2idx['stat']:
            batch['stat_input_ids'].append(torch.tensor(self.divide_into_chunks(self.stat_textmap[s]), dtype=torch.long))
        
        
        for k in ['query', 'stat']:
            batch[f'{k}_input_ids'] = pad_sequence(batch[f'{k}_input_ids'], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            batch[f'{k}_attention_mask'] = batch[f'{k}_input_ids'] != self.tokenizer.pad_token_id
            if k == 'query':
                batch[f'{k}_pos'] = torch.tensor(batch[f'{k}_pos'], dtype=torch.long)
            
        # batch['stat_relevance_labels'] = torch.zeros_like(batch['stat_pos'])
        # batch['stat_relevance_labels'][:, 0] = 1
        # batch['prec_relevance_labels'] = torch.zeros_like(batch['prec_pos'])
        # batch['prec_relevance_labels'][:, 0] = 1
        
        batch['stat_relevance_labels'] = torch.zeros(len(queries), len(cand2idx['stat']))
        batch['stat_relevance_mask'] = torch.ones_like(batch['stat_relevance_labels']).bool()
        for i,qry in enumerate(queries):
            for s in qry['stat_all_relevant']:
                if s == qry['stat_relevant']: batch['stat_relevance_labels'][i, cand2idx['stat'][s]] = 1
                elif s in cand2idx['stat']: batch['stat_relevance_mask'][i, cand2idx['stat'][s]] = False
        return BatchEncoding(batch)
    
    
    
@dataclass
class SAILERForInferenceDataCollator:
    def __init__(self, tokenizer, data_type, stat_embeddings=None, chunk_size=512, overlap_size=128):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.stat_embeddings = stat_embeddings
        self.data_type = data_type
        
    def divide_into_chunks(self, text):
        i = 0
        chunks = []
        while i < len(text):
            _text = text[i:i+self.chunk_size-2]
            chunk = [self.tokenizer.cls_token_id] + _text + [self.tokenizer.sep_token_id]
            chunk += [self.tokenizer.pad_token_id] * (self.chunk_size - len(chunk))
            chunks.append(chunk)
            i += self.chunk_size - self.overlap_size
        return chunks
    
    def single_chunk(self, text):
        chunk = [self.tokenizer.cls_token_id] + text[:self.chunk_size-2] + [self.tokenizer.sep_token_id]
        chunk += [self.tokenizer.pad_token_id] * (self.chunk_size - len(chunk))
        return [chunk]
    
    def __call__(self, examples):
        batch = {}
        for k in ['query', 'stat']:
            if k != self.data_type: continue
            batch[f'{k}_ids'] = []
            batch[f'{k}_input_ids'] = []
            
            for exp in examples:
                if k == 'stat': id2idx = stat2idx
                else: id2idx = qry2idx
                batch[f'{k}_ids'].append(id2idx[exp['id']])
                batch[f'{k}_input_ids'].append(torch.tensor(self.divide_into_chunks(exp['input_ids']), dtype=torch.long))
            
            batch[f'{k}_ids'] = torch.tensor(batch[f'{k}_ids'], dtype=torch.long)
            batch[f'{k}_input_ids'] = pad_sequence(batch[f'{k}_input_ids'], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            batch[f'{k}_attention_mask'] = batch[f'{k}_input_ids'] != self.tokenizer.pad_token_id
            
            if k == 'query':
                batch['stat_embeddings'] = self.stat_embeddings
            return BatchEncoding(batch)

@dataclass
class SAILEROutput(ModelOutput):
    loss:torch.Tensor = None
    stat_scores:torch.Tensor = None
    query_ids:List = None
    stat_ids:List = None
    
class SAILER(nn.Module):
    def __init__(self, 
                 bert_model:Union[str, BertModel],
                 loss_type:str='categorical'
                 ):
        super().__init__()
        
        # if type(bert_model) == str:
        self.qry_bert_model = AutoModel.from_pretrained(bert_model)
        self.stat_bert_model = AutoModel.from_pretrained(bert_model)
        # self.prec_bert_model = self.qry_bert_model
        
        # for bert in [self.qry_bert_model, self.stat_bert_model, self.prec_bert_model]:
        #     for n,p in bert.named_parameters():
        #         if not n.startswith("encoder.layer.11") and not n.startswith("encoder.layer.10") and not n.startswith("pooler"):
        #             p.requires_grad = False    
            
        # self.bert_model = bert_model
        # self.pooler = nn.Linear(768, 768)
        self.hidden_size = self.qry_bert_model.config.hidden_size
        self.loss = nn.CrossEntropyLoss() if loss_type=='categorical' else nn.BCEWithLogitsLoss()
        self.dropout = nn.Dropout(0.1)
        
        self.gelu = nn.GELU()
        self.cos = nn.CosineSimilarity(dim=-1)
        
        self.skipped = 0
        
        
    def encode(self,
                bert_model:BertModel=None,
                input_ids:torch.Tensor=None, # [num inputs, max segs, max seg len]
                attention_mask:torch.Tensor=None, # [num inputs, max segs, max seg len]
                encoding_segments:Optional[int]=None
               ):
        ni, ms, msl = input_ids.shape
        input_ids_flat = input_ids.view(-1, msl) # [num inputs * max segs, max seg len]
        attention_mask_flat = attention_mask.view(-1, msl) # [num inputs * max segs, max seg len]
        valid_mask = attention_mask_flat.any(dim=-1) # [num inputs * max segs,]
        
        input_ids_flat_valid = input_ids_flat[valid_mask, :] # [num valid segs, max seg len]
        attention_mask_flat_valid = attention_mask_flat[valid_mask, :] # [num valid segs, max seg len]
        
        cls_outputs_flat = torch.zeros(ni * ms, self.hidden_size, device=input_ids.device) # [num inputs * max segs, hidden dim]
        
        #print("***", input_ids.shape, input_ids_flat_valid.shape)
        if encoding_segments is None:
            cls_outputs_flat_valid = self.dropout(bert_model(input_ids=input_ids_flat_valid, attention_mask=attention_mask_flat_valid).last_hidden_state[:, 0, :])
        else:
            cls_outputs_flat_valid = []
            for j in range(0, input_ids_flat_valid.size(0), encoding_segments):
                # print(f"+++{j}")
                input_ids_flat_valid_segment = input_ids_flat_valid[j : j + encoding_segments]
                attention_mask_flat_valid_segment = attention_mask_flat_valid[j : j + encoding_segments]
                cls_outputs_flat_valid_segment = self.dropout(bert_model(input_ids=input_ids_flat_valid_segment, attention_mask=attention_mask_flat_valid_segment).last_hidden_state[:, 0, :])
                cls_outputs_flat_valid.append(cls_outputs_flat_valid_segment)
            cls_outputs_flat_valid = torch.cat(cls_outputs_flat_valid, dim=0)
        
        cls_outputs_flat[valid_mask, :] = cls_outputs_flat_valid
        cls_outputs = cls_outputs_flat.view(ni, ms, self.hidden_size) # [num inputs, max segs, hidden dim]
        
        # return cls_outputs.sum(dim=1) / attention_mask.any(dim=-1).float().sum(dim=-1).unsqueeze(1) # [num 
        # inputs, hidden dim]
        
        doc_outputs = [cls_outputs[i, :attention_mask[i].any(dim=-1).long().sum(), :].mean(dim=0) for i in range(cls_outputs.size(0))]
        # doc_outputs = [cls_outputs[i].mean(dim=0) for i in range(cls_outputs.size(0))]
        return torch.stack(doc_outputs, dim=0)
                  
        
    # def encode(self, input_ids, attention_mask, encoding_segments=None):
    #     return self.pooler(self._encode(input_ids, attention_mask, encoding_segments=encoding_segments))
    #     # return self._encode(input_ids, attention_mask, encoding_segments=encoding_segments)
    def gradient_checkpointing_enable(self):
        self.stat_bert_model.gradient_checkpointing_enable()
        self.stat_bert_model.enable_input_require_grads()
        self.qry_bert_model.gradient_checkpointing_enable()
        self.qry_bert_model.enable_input_require_grads()
        
    def forward(self, 
                query_pos: torch.Tensor = None, # [num qrys,]
                # stat_pos: torch.Tensor = None, # [num qrys, num cands]
                # prec_pos: torch.Tensor = None, # [num qrys, num cands/2]
                query_input_ids: torch.Tensor = None, # [num qrys, max qry segs, seg size]
                stat_input_ids: torch.Tensor = None, # [num stats, max stat segs, seg size]
                query_attention_mask: torch.Tensor = None, # [num qrys, max qry segs, seg size] 
                stat_attention_mask: torch.Tensor = None, # [num stats, max stat segs, seg size]
                stat_relevance_labels: torch.Tensor = None, # [num qrys, num stats]
                stat_relevance_mask: torch.Tensor = None, # [num qrys, num stats]
                ):
        
        
        query_encoded = self.encode(self.qry_bert_model, query_input_ids, query_attention_mask, encoding_segments=None) # [num qrys, hidden dim]
    
        stats_encoded = self.encode(self.stat_bert_model, stat_input_ids, stat_attention_mask, encoding_segments=None) # [num stats, hidden dim]

               
        # # No in-batch sampling
        # query_encoded = query_encoded[query_pos] # 
        # stats_encoded = torch.gather(stats_encoded.unsqueeze(0).repeat(query_encoded.size(0), 1, 1), 1, stat_pos.unsqueeze(2).repeat(1, 1, stats_encoded.size(1)))
        # precs_encoded = torch.gather(precs_encoded.unsqueeze(0).repeat(query_encoded.size(0), 1, 1), 1, prec_pos.unsqueeze(2).repeat(1, 1, precs_encoded.size(1)))
        
        # stat_scores = torch.bmm(query_encoded.unsqueeze(1), stats_encoded.permute(0,2,1)).squeeze(1) # [num qrys, num stats]
        # stat_loss = self.loss(stat_scores, stat_relevance_labels.argmax(dim=1))
        # prec_scores = torch.bmm(query_encoded.unsqueeze(1), precs_encoded.permute(0,2,1)).squeeze(1) # [num qrys, num precs]
        # prec_loss = self.loss(prec_scores, prec_relevance_labels.argmax(dim=1))
        
        # In-batch sampling
        query_encoded = query_encoded[query_pos] # [num all qrys, hidden dim]
        stat_scores = torch.matmul(query_encoded, stats_encoded.permute(1,0))
        stat_scores.masked_fill_(~stat_relevance_mask, -1e2)     
        
        loss = self.loss(stat_scores, stat_relevance_labels)
        
        if loss == torch.nan:
            print("NaN found")
            
        return SAILEROutput(loss=loss, stat_scores=stat_scores, query_ids=None, stat_ids=None)

@dataclass
class SAILERForInferenceOutput(ModelOutput):
    stat_ids:List = None
    stat_embeddings:torch.Tensor=None
    query_ids:List = None
    query_embeddings:torch.Tensor=None
    stat_scores:torch.Tensor = None

class SAILERForInference(SAILER):
    def __init__(self, 
                 bert_model:Union[str, BertModel],
                 ):
        super().__init__(bert_model)
        self.cos = nn.CosineSimilarity(dim=-1)
    
    @torch.no_grad()    
    def encode(self, bert_model, input_ids, attention_mask, encoding_segments=None):
        return super(SAILERForInference, self).encode(bert_model, input_ids, attention_mask, encoding_segments)
        
    def forward(self, 
                query_ids: List = None, # [num qrys,]
                stat_ids: List = None, # [num stats,]
                query_input_ids: torch.Tensor = None, # [num qrys, max qry segs, seg size]
                stat_input_ids: torch.Tensor = None, # [num stats, max stat segs, seg size]              
                query_attention_mask: torch.Tensor = None, # [num qrys, max qry segs, seg size] 
                stat_attention_mask: torch.Tensor = None, # [num stats, max stat segs, seg size]
                stat_embeddings: torch.Tensor = None, # [num stats, hidden dim]
                ):
        
        if stat_ids is not None:
            stats_encoded = self.encode(self.stat_bert_model, stat_input_ids, stat_attention_mask)
            return SAILERForInferenceOutput(stat_ids=stat_ids, stat_embeddings=stats_encoded)
        
        if query_ids is not None:
            # print(stat_embeddings.shape)
            query_encoded = self.encode(self.qry_bert_model, query_input_ids, query_attention_mask)
            
            # query_expanded = query_encoded.unsqueeze(1).repeat(1,stat_embeddings.size(0),1)
            # stat_expanded = stat_embeddings.unsqueeze(0).repeat(query_encoded.size(0), 1, 1)
            # stat_scores = self.cos(query_expanded, stat_expanded)
            
            # query_expanded = query_encoded.unsqueeze(1).repeat(1,prec_embeddings.size(0),1)
            # prec_expanded = prec_embeddings.unsqueeze(0).repeat(query_encoded.size(0), 1, 1)
            # prec_scores = self.cos(query_expanded, prec_expanded)
            stat_scores = torch.matmul(query_encoded, stat_embeddings.permute(1,0))
            return SAILERForInferenceOutput(query_ids=query_ids, stat_scores=stat_scores)
           

accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=GRAD_ACC)
# accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=4)
device = accelerator.device
         
if TRAIN_DEV:             

    sailer_collator = SAILERDataCollator(stat_tok_textmap, tokenizer, loss_type='categorical', cands_per_qry=NUM_CANDS)
    model = SAILER('CSHaitao/SAILER_en_finetune')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_params)

    # for name, param in model.named_parameters():
    #     if 'encoder.layer.10' not in name and 'encoder.layer.11' not in name:
    #         param.requires_grad = False
            
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(trainable_params)
    
    # model.load_state_dict(torch.load("small/sailer-ft-tied/pytorch_model.bin", map_location='cuda'), strict=False)
    optimizer = AdamW(model.parameters(), lr=5e-6)
    # optimizer = AdamW(
    #     [
    #         {'params': list(model.qry_bert_model.parameters()) + list(model.prec_bert_model.parameters()), 'lr': 5e-6},
    #         {'params': model.stat_bert_model.parameters(), 'lr': 5e-6},
    #         # {'params': model.prec_bert_model.parameters(), 'lr': 5e-6},
    #         # {'params': model.pooler.parameters(), 'lr': 1e-3}
    #     ], lr=3e-5)
    num_steps = math.ceil(len(train_dataset) / (BATCH_SIZE * GRAD_ACC)) * EPOCHS
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_steps/10), num_training_steps=int(num_steps * 1.2)) 

    model.to(device)
    model.gradient_checkpointing_enable()
    

    # dev_dataset = torch.utils.data.Subset(dev_dataset, list(range(300)))
    # test_dataset = torch.utils.data.Subset(test_dataset, list(range(300)))

    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=sailer_collator, shuffle=True) #, num_workers=8)
    model, optimizer, train_dl, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dl, lr_scheduler
    )
    # print(model)

    dev_dl = DataLoader(dev_dataset, batch_size=BATCH_SIZE*4, collate_fn=sailer_collator, shuffle=False) #, num_workers=8)
    dev_dl = accelerator.prepare(dev_dl)

    best_loss = torch.inf
    # best_loss = 1.8652063531753345
    best_model = model.state_dict()
    LOG = []

    print('\n', num_steps, '\n')    

    for epoch in range(0, EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        train_loss = 0
        num_train_batches = 0
        skipped_for = 0
        skipped_back = 0
        
        for k,batch in enumerate(tqdm(train_dl, desc="Training...")):
            torch.cuda.empty_cache()
            model.train()
            
            with accelerator.accumulate(model):
                try:
                    output = model(**batch)
                except torch.cuda.OutOfMemoryError:
                    skipped_for += 1
                    print("Skipped forward", skipped_for) #, batch['stat_relevance_labels'].shape, batch['prec_relevance_labels'].shape)
                    continue
                    
                try:    
                    accelerator.backward(output.loss)
                except torch.cuda.OutOfMemoryError:
                    skipped_back += 1
                    print("Skipped backward", skipped_back) #, batch['stat_relevance_labels'].shape, batch['prec_relevance_labels'].shape)
                    del output
                    continue
                
                train_loss += output.loss.item() / GRAD_ACC
                num_train_batches += 1
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                    
            if (k + 1)%int(num_steps * GRAD_ACC/(10 * EPOCHS)) == 0:
                torch.cuda.empty_cache()  
                model.eval()
                test_loss = 0
                num_test_batches = 0
                skipped_test = 0
                with torch.no_grad():
                    for batch in tqdm(dev_dl, desc="Evaluating..."):
                        torch.cuda.empty_cache()
                        
                        try:
                            output = model(**batch)
                        except torch.cuda.OutOfMemoryError:
                            skipped_test += 1
                            continue
                        
                        test_loss += output.loss.item()
                        num_test_batches += 1

                TR_loss = train_loss/num_train_batches
                DV_loss = test_loss/num_test_batches
                
                if DV_loss < best_loss:
                    best_loss = DV_loss
                    best_model = model.state_dict()
                    torch.save(best_model, f"/home/shounak/HDD/Layman-LSI/model/sailer-{SUFFIX}/pytorch_model.bin")
            
                print(f"Epoch: {epoch:3d} | Tr. Loss: {TR_loss:.4f} | Tr. Skip: {skipped_for+skipped_back:6d} | Dv. Loss: {DV_loss:.4f} | Dv. Skip: {skipped_test:6d}")
                
                LOG.append((epoch, TR_loss, DV_loss))
                with open(f"/home/shounak/HDD/Layman-LSI/model/sailer-{SUFFIX}/log.json", 'w') as fw:
                    json.dump(LOG, fw, indent=4)
                    
                train_loss = 0
                num_train_batches = 0
                skipped_for = 0
                skipped_back = 0

if TEST:
    print("LOADING MODEL ...")      
    model_inf = SAILERForInference('CSHaitao/sailer_en_finetune').cuda()
    # model_inf.load_state_dict(torch.load(f"/home/shounak/HDD/Layman-LSI/model/sailer-{SUFFIX}/pytorch_model.bin", map_location='cuda'), strict=False)
    model_inf.eval()
    model_inf.to(device)
    model_inf = accelerator.prepare(model_inf)


    if os.path.exists(f"/home/shounak/HDD/Layman-LSI/model/sailer-{SUFFIX}/statute-embeddings.pt"):
        stat_ids, stat_embeddings = torch.load(f"/home/shounak/HDD/Layman-LSI/model/sailer-{SUFFIX}/statute-embeddings.pt")
    else:
        sailer_test_stat_collator = SAILERForInferenceDataCollator(tokenizer, data_type='stat', chunk_size=512, overlap_size=128)
        test_stat_dl = DataLoader(stat_dataset, batch_size=32, collate_fn=sailer_test_stat_collator, num_workers=1)
        test_stat_dl = accelerator.prepare(test_stat_dl)
        stat_ids = []
        stat_embeddings = []
        for batch in tqdm(test_stat_dl, desc="Encoding statutes"):
            torch.cuda.empty_cache()
            for k,v in batch.items():
                if type(v) == torch.Tensor:
                    batch[k]=v.cuda()
            output = model_inf(**batch)
            stat_ids.extend(output.stat_ids)
            stat_embeddings.append(output.stat_embeddings.detach().cpu())
            del output
        stat_embeddings = torch.cat(stat_embeddings, dim=0)
        print(len(stat_ids), stat_embeddings.shape)
        torch.save([stat_ids, stat_embeddings], f"/home/shounak/HDD/Layman-LSI/model/sailer-{SUFFIX}/statute-embeddings.pt")

    print(stat_embeddings.shape)
    
    final_stat_embeddings = torch.zeros(stat_embeddings.shape)
    
    for i,s in enumerate(stat_ids):
        # final_stat_embeddings[stat2idx[s]] = stat_embeddings[i]
        final_stat_embeddings[s] = stat_embeddings[i]
    
    
    sailer_test_collator = SAILERForInferenceDataCollator(tokenizer, data_type='query', stat_embeddings=final_stat_embeddings, chunk_size=512, overlap_size=128)
    test_dl = DataLoader(test_dataset, batch_size=16, collate_fn=sailer_test_collator)
    # test_dl = DataLoader(train_dataset+dev_dataset, batch_size=16, collate_fn=sailer_test_collator)
    # test_dl = DataLoader(dev_dataset, batch_size=4, collate_fn=sailer_test_collator)
    # test_dl = accelerator.prepare(test_dl)
    qry_ids = []
    stat_scores = []
    for batch in tqdm(test_dl, desc="Encoding queries"):
        torch.cuda.empty_cache()
        for k,v in batch.items():
            if type(v) == torch.Tensor:
                batch[k]=v.cuda()
        output = model_inf(**batch)
        qry_ids.extend(output.query_ids)
        stat_scores.append(output.stat_scores)
    stat_scores = torch.cat(stat_scores, dim=0)
    
    final_stat_scores = torch.zeros(stat_scores.shape)
    for i,q in enumerate(qry_ids):
        # final_stat_scores[qry2idx[q]] = stat_scores[i]
        # final_prec_scores[qry2idx[q]] = prec_scores[i]
        final_stat_scores[q] = stat_scores[i]
    
    print(final_stat_scores.shape)
    torch.save(final_stat_scores, f"/home/shounak/HDD/Layman-LSI/model/sailer-{SUFFIX}/secs-scores.pt")
    # torch.save(final_stat_scores, f"/home/shounak/HDD/Layman-LSI/model/sailer-{SUFFIX}/secs-scores-traindev.pt")