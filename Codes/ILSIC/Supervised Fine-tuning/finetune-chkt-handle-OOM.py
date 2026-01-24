import os
import torch
import pandas as pd
import textwrap
import numpy as np
import re
import nltk
import json
import time

from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed,
    BitsAndBytesConfig, TrainerCallback, AutoConfig
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from huggingface_hub import login

def build_prompt_llama3(tokenizer, source, target):
    system_message = (
        """
        You are a legal expert on Indian law. Given the text below, identify all applicable legal sections and their full act names. Follow these rules exactly:
        1. Identify every relevant provision (section number + full act name) you are 100% certain applies under Indian statutes.
        2. Output only one list in square brackets, formatted like: ["Section X of Act Name"; "Section Y of Act Name"; …]
            - Each entry must be a quoted string with the section first, then the full act name.
            - Each entry needs both section number AND full act name
            - Separate entries with a semicolon.
            - Do not include anything outside this single list.
        3. Exclude any entry missing either section or act (no incomplete pairs).
        4. Do not repeat identical section-act pairs (no duplicates).
        5. Do not add explanations, labels, or extra text—only the list itself.
        6. Always spell out the full name of each act (no abbreviations).
        7. Include only provisions that are clearly and directly relevant (no speculative or uncertain entries).
        """
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": source},
        {"role": "assistant", "content": target}
    ]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return chat_prompt.strip()

def build_prompt_gemma3(tokenizer, source, target):
    system_message = (
        """
        You are a legal expert on Indian law. Given the text below, identify all applicable legal sections and their full act names. Follow these rules exactly:
        1. Identify every relevant provision (section number + full act name) you are 100% certain applies under Indian statutes.
        2. Output only one list in square brackets, formatted like: ["Section X of Act Name"; "Section Y of Act Name"; …]
            - Each entry must be a quoted string with the section first, then the full act name.
            - Each entry needs both section number AND full act name
            - Separate entries with a semicolon.
            - Do not include anything outside this single list.
        3. Exclude any entry missing either section or act (no incomplete pairs).
        4. Do not repeat identical section-act pairs (no duplicates).
        5. Do not add explanations, labels, or extra text—only the list itself.
        6. Always spell out the full name of each act (no abbreviations).
        7. Include only provisions that are clearly and directly relevant (no speculative or uncertain entries).
        """
    )
    user_message = f"{system_message}\n\n{source}"
    messages = [
        {"role": "user", "content": user_message},
        {"role": "model", "content": target}
    ]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return chat_prompt.strip()

def get_prompt_builder(model_id):
    if "llama" in model_id.lower():
        return build_prompt_llama3
    elif "gemma" in model_id.lower():
        return build_prompt_gemma3
    else:
        raise ValueError("Unknown model type (only supports Llama-3 and Gemma-3)")


if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print(f"Number of GPUs available: {torch.cuda.device_count()}")

    nltk.download("punkt", quiet=True)

    login(token="Hugging face api")  

    model_short_name = "Llama_3_SynLM"
    task_name = "SynLM_LSI_train"
    bt_name = 1
    lr_name = "1e5"
    lang = 'English'
    seed_val = 42
    OVERLONG_INDICES_PATH = "/home/shounak/HDD/Layman-LSI/model/Overlong_indices/gemma_lm_ft_1.json"
    
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_id = "google/gemma-3-12b-it"

    LONG_CONTEXT_WINDOW = 32000 

    set_seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    def format_target_list(target):
        return "[" + "; ".join(f'"{item}"' for item in target) + "]"

    def whitespace_handler(text):
        text = re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', text.strip()))
        text = text.replace("\xad", "")
        return text

    def load_local_dataset(dataset_path):
        df = pd.read_json(dataset_path, orient='records', lines=True, compression='xz')
        df["target"] = df["answer"].apply(format_target_list)
        df["source"] = df["instruction"]
        print(f"Shape of {dataset_path}: {df.shape}")
        df = df[["source", "target"]]
        return df

    train_path = "/home/shounak/HDD/Layman-LSI/dataset/court-dataset/FT-Court-train.jsonl.xz"
    val_path = "/home/shounak/HDD/Layman-LSI/dataset/court-dataset/FT-Court-dev.jsonl.xz"
    test_path = "/home/shounak/HDD/Layman-LSI/dataset/court-dataset/FT-Court-test.jsonl.xz"
    
    train_df = load_local_dataset(train_path)
    val_df = load_local_dataset(val_path)
    test_df = load_local_dataset(test_path)

    print(f"Train df shape: {train_df.shape}")
    print(f"Validation df shape: {val_df.shape}")
    print(f"Test df shape: {test_df.shape}")

    sample_data = val_df.iloc[12]
    print("source:")
    for wrp in textwrap.wrap(sample_data['source'], width=100):
        print(whitespace_handler(wrp))
    print("============")
    print("target:")
    target_text = sample_data['target']
    for wrp in textwrap.wrap(target_text, width=100):
        print(whitespace_handler(wrp))

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    print(f"Train dataset size: {len(dataset_dict['train'])}")
    print(f"Validation dataset size: {len(dataset_dict['validation'])}")
    print(f"Test dataset size: {len(dataset_dict['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
        model_max_length=LONG_CONTEXT_WINDOW 
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = LONG_CONTEXT_WINDOW 

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Padding side: {tokenizer.padding_side}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print(f"eos_token: {tokenizer.eos_token}")

    prompt_builder = get_prompt_builder(model_id)
    def apply_template_decoder_type_chat(dataset, lang):
        def map_fn(exm):
            source = whitespace_handler(exm['source']).strip()
            target = whitespace_handler(exm['target']).strip()
            prompt = prompt_builder(tokenizer, source, target)
            ex = {"prompt": prompt}
            return ex

        def filter_fn(exm):
            return len(exm["source"]) > 0 and len(exm["target"]) > 0

        dataset = dataset.map(lambda x: map_fn(x)).filter(filter_fn)
        return dataset

    dataset_dict["train"] = apply_template_decoder_type_chat(dataset_dict["train"], lang)
    dataset_dict["validation"] = apply_template_decoder_type_chat(dataset_dict["validation"], lang)
    dataset_dict["test"] = apply_template_decoder_type_chat(dataset_dict["test"], lang)

    for split in ["train", "validation", "test"]:
        dataset_dict[split] = dataset_dict[split].rename_column("prompt", "text")

    print(dataset_dict["validation"][12]["text"])  
    
    
    split_to_file = {
        "train": train_path,
        "validation": val_path,
        "test": test_path,
    }
    overlong_indices_info = {}

    for split in ["train", "validation", "test"]:
        dataset = dataset_dict[split]
        indices = []
        for i, example in enumerate(dataset):
            n_tokens = len(tokenizer(example["text"])["input_ids"])
            if n_tokens > tokenizer.model_max_length:
                indices.append(i)
        overlong_indices_info[split] = {
            "source_file": split_to_file[split],
            "indices": indices
        }
        print(f"{split}: Found {len(indices)} overlong examples (will be saved in {OVERLONG_INDICES_PATH}).")

    with open(OVERLONG_INDICES_PATH, "w") as f:
        json.dump(overlong_indices_info, f, indent=2)
    print(f"Saved overlong indices info to {OVERLONG_INDICES_PATH}")
    
    def filter_overlong_examples(example):
        n_tokens = len(tokenizer(example["text"])["input_ids"])
        return n_tokens <= tokenizer.model_max_length

    for split in ["train", "validation", "test"]:
        dataset_dict[split] = dataset_dict[split].filter(filter_overlong_examples)
    
    max_percentile = 95
    tokenized_inputs = concatenate_datasets([
        dataset_dict["train"],
        dataset_dict["validation"],
        dataset_dict["test"]
    ]).map(
        lambda x: tokenizer(x["text"], truncation=False, max_length=LONG_CONTEXT_WINDOW),
        batched=True,
        remove_columns=["source", "target"]
    )

    input_lengths = [len(x) for x in tokenized_inputs["input_ids"]]
    max_prompt_length = max(input_lengths)
    avg_prompt_length = sum(input_lengths) / len(input_lengths)
    max_prompt_percentile = int(np.percentile(input_lengths, max_percentile))
    print(f"Max prompt length: {max_prompt_length}")
    print(f"Avg prompt length: {avg_prompt_length}")
    print(f"Max prompt percentile of {max_percentile}: {max_prompt_percentile}")

    bits_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    config = AutoConfig.from_pretrained(model_id)
    config.model_max_length=LONG_CONTEXT_WINDOW
    config.max_position_embeddings = LONG_CONTEXT_WINDOW

    model_ori = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config, 
        quantization_config=bits_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir="/home/shared/hf_cache"
    )
    
    print("max_position_embeddings", model_ori.config.max_position_embeddings)
    print("tokenizer.model_max_length", tokenizer.model_max_length)

    model_ori.config.use_cache = False
    model_ori.gradient_checkpointing_enable()
    model_ori = prepare_model_for_kbit_training(model_ori)
    model_ori = get_peft_model(model_ori, lora_config)

    def get_data_collator(model_id, tokenizer):
        if "llama" in model_id.lower():
            return DataCollatorForCompletionOnlyLM("<|start_header_id|>assistant<|end_header_id|>\n\n", tokenizer=tokenizer, mlm=False)
        elif "gemma" in model_id.lower():
            return DataCollatorForCompletionOnlyLM("<start_of_turn>model\n", tokenizer=tokenizer, mlm=False)
        else:
            raise ValueError("Unknown model type (only supports Llama-3 and Gemma-3)")

    collator = get_data_collator(model_id, tokenizer)

    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 0.00001
    OUTPUT_DIR = f"/home/shounak/HDD/Layman-LSI/model/{model_short_name}_FT_bt_{bt_name}_lr_{lr_name}"
    N_EPOCHS = 5
    log_run_name = f"{model_short_name}_{task_name}_log_run_bt_{bt_name}_lr_{lr_name}"

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        bf16=True,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        weight_decay=0.01,
        save_safetensors=True,
        lr_scheduler_type="cosine",
        num_train_epochs=N_EPOCHS,
        logging_dir=f"{OUTPUT_DIR}/logs/{log_run_name}",
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",       
        report_to="tensorboard",
        run_name=f"{log_run_name}",
        seed=42
    )

    class EvalLossAndTimeRecorderCallback(TrainerCallback):
        def __init__(self, output_path):
            self.output_path = output_path
            self.loss_history = []
            self.epoch_start_time = None

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.epoch_start_time = time.time()

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            epoch_end_time = time.time()
            epoch_duration = None
            if self.epoch_start_time is not None:
                epoch_duration = epoch_end_time - self.epoch_start_time
            entry = {
                "epoch": float(state.epoch) if state.epoch is not None else None,
                "global_step": int(state.global_step),
                "eval_loss": float(metrics.get("eval_loss", -1)),
                "epoch_time_sec": float(epoch_duration) if epoch_duration is not None else None
            }
            self.loss_history.append(entry)
            with open(self.output_path, "w") as f:
                json.dump(self.loss_history, f, indent=2)
            self.epoch_start_time = None  

        def on_train_end(self, args, state, control, **kwargs):
            with open(self.output_path, "w") as f:
                json.dump(self.loss_history, f, indent=2)
                
                
    class EpochProgressCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            total_epochs = int(getattr(args, "num_train_epochs", 0))
            current_epoch = int(state.epoch) + 1 if state.epoch is not None else 1
            print(f"\n--- Starting epoch {current_epoch}/{total_epochs} ---\n")
    
    class RobustOOMSkippingSFTTrainer(SFTTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.oom_batches = []

        def training_step(self, model, inputs, num_items=None):
            try:
                return super().training_step(model, inputs, num_items)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_idx = self.state.global_step
                    print(f"OOM at batch {batch_idx}, skipping!")
                    self.oom_batches.append(batch_idx)
                    torch.cuda.empty_cache()
                    return torch.tensor(0.0, device=self.args.device, requires_grad=True)
                else:
                    raise e


        def save_oom_log(self, path):
            with open(path, "w") as f:
                json.dump(self.oom_batches, f, indent=2)

    trainer = RobustOOMSkippingSFTTrainer(
        model=model_ori,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        peft_config=lora_config,
        data_collator=collator,
        args=training_args,
        callbacks=[
            EvalLossAndTimeRecorderCallback(output_path=f"{OUTPUT_DIR}/eval_loss_history.json"),
            EpochProgressCallback(),
        ]
    )

    trainer.callback_handler.callbacks = [
        cb for cb in trainer.callback_handler.callbacks
        if cb.__class__.__name__ != "SFTPrintingCallback"
    ]

    trainer.train()

    if hasattr(trainer, "oom_batches"):
        oom_log_path = os.path.join(OUTPUT_DIR, "oom_batches.json")
        trainer.save_oom_log(oom_log_path)
        print(f"Saved OOM batch indices (if any) to {oom_log_path}")

    LORA_FT = True

    if LORA_FT:
        peft_id = "Q_LoRA_Model"
        output_model_id = OUTPUT_DIR
        peft_model_id = os.path.join(output_model_id, peft_id)

        if not os.path.exists(peft_model_id):
            os.makedirs(peft_model_id)

        trainer.model.save_pretrained(peft_model_id)
        tokenizer.save_pretrained(peft_model_id)

    else:
        trainer.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

    OUTPUT_DIR_NEW = f"/home/shounak/HDD/Layman-LSI/model/{OUTPUT_DIR}_final"
    if not os.path.exists(OUTPUT_DIR_NEW):
        os.makedirs(OUTPUT_DIR_NEW)

    trainer.save_model(OUTPUT_DIR_NEW)
    tokenizer.save_pretrained(OUTPUT_DIR_NEW)

    print(trainer.model)
    