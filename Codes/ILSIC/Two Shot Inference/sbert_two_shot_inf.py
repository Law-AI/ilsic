import os, json, time, re
from typing import List, Dict, Any, Tuple, Optional

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from openai import OpenAI
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_SEQ_LENGTH = 32000
TEMPERATURE = 0.1
SAVE_EVERY = 10

"""
How to toggle adapter usage:
- Base model only:
    USE_ADAPTER = False
- Base + LoRA/PEFT adapter:
    USE_ADAPTER = True
    ADAPTER_PATH = "/path/to/peft/adapter"
- Merging adapter into base:
    MERGE_ADAPTER = True   # Only viable when NOT using 4-bit quantization.
    (With 4-bit quantization, keep MERGE_ADAPTER = False.)
"""
USE_ADAPTER   = False
ADAPTER_PATH  = "/home/shounak/HDD/Layman-LSI/model/Gemma_3_LM_FT_bt_1_lr_1e5/checkpoint-4851"
MERGE_ADAPTER = False  

"""
How to toggle usage:
- Local HF model (Llama/Gemma):
    USE_OPENAI = False
    USE_ADAPTER = True/False (depending on if you want adapter)
- GPT-4.1 via OpenAI:
    USE_OPENAI = True
    OPENAI_API_KEY = "your_key"
"""
USE_OPENAI   = False  # <<< toggle this to True when you want GPT-4.1
OPENAI_MODEL = "gpt-4.1"
OPENAI_API_KEY = "OpenAI API Key"   # <<< put your real key here

# Data paths
QUERIES_JSON_PATH  = "/home/shounak/HDD/Layman-LSI/dataset/Layman-new-dataset/id-Layman-test-clean.json"
EXAMPLES_JSON_PATH = "/home/shounak/HDD/Layman-LSI/dataset/Layman-new-dataset/id-Layman-train-clean.json"
OUTPUT_JSON_PATH   = "/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/two-shot/base_gpt4-1_two_shot.json"

SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CATS_REQUIRE_RELIGION = {"Family Law", "Property Law", "Civil Law"}


INSTRUCTION_BLOCK = (
    "You are a legal expert on Indian law. Given the text below, identify all applicable legal sections and their full act names.\n"
    "Follow these rules exactly:\n"
    "1. Identify every relevant provision (section number + full act name) you are 100% certain applies under Indian statutes.\n"
    "2. Output only one list in square brackets, formatted like: [\"Section X of Act Name\"; \"Section Y of Act Name\"; …]\n"
    "   - Each entry must be a quoted string with the section first, then the full act name.\n"
    "   - Each entry needs both section number AND full act name.\n"
    "   - Separate entries with a semicolon.\n"
    "   - Do not include anything outside this single list.\n"
    "3. Exclude any entry missing either section or act (no incomplete pairs).\n"
    "4. Do not repeat identical section-act pairs (no duplicates).\n"
    "5. Do not add explanations, labels, or extra text—only the list itself.\n"
    "6. Below are two EXAMPLES of the task (input 1/2 with output1/2). Then produce the answer ONLY for input 3 by writing output3. "
    "Stop immediately after you close the single list with a ']'"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

if USE_OPENAI:
    print(f"[Init] Using OpenAI model: {OPENAI_MODEL}")
    client = OpenAI(api_key=OPENAI_API_KEY)
    tokenizer = None
    model = None
else:
    try:
        from peft import PeftModel
        PEFT_AVAILABLE = True
    except Exception:
        PEFT_AVAILABLE = False

    print("[Init] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    print(f"[Init] Loading BASE model: {MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=bnb_config,
        cache_dir="/home/shared/hf_cache",
    )

    if USE_ADAPTER:
        if not ADAPTER_PATH:
            raise ValueError("USE_ADAPTER=True but ADAPTER_PATH is empty.")
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft not installed but USE_ADAPTER=True.")
        print(f"[Init] Attaching PEFT adapter from: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        if MERGE_ADAPTER:
            try:
                model = model.merge_and_unload()
                print("[Init] Merge successful; adapter unloaded.")
            except Exception as e:
                print(f"[Warn] Merge failed: {e}\n       Continuing with adapter active.")
    else:
        print("[Init] Using BASE model only (no adapter).")
        model = base_model

print("[Init] Loading SBERT for similarity …")
sbert = SentenceTransformer(SBERT_MODEL_NAME, device=device)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def format_full_citations_as_list(full_citations: List[str]) -> str:
    quoted = [f"\"{c.strip()}\"" for c in full_citations if c and c.strip()]
    return "[" + "; ".join(quoted) + "]"

def maybe_religion_line(category: str, religion_value: Optional[str]) -> str:
    if category in CATS_REQUIRE_RELIGION:
        rv = (religion_value or "Unknown").strip()
        return f"\nquery-religion: {rv}"
    return ""

def build_two_shot_prompt(
    instruction: str,
    cat_for_prompt: str,
    ex1_text: str, ex1_religion: Optional[str], ex1_full_cits: List[str],
    ex2_text: str, ex2_religion: Optional[str], ex2_full_cits: List[str],
    input3_text: str, input3_religion: Optional[str],
) -> str:
    return (
        f"{instruction}\n\n"
        f"input 1: {ex1_text}{maybe_religion_line(cat_for_prompt, ex1_religion)}\n"
        f"output1: {format_full_citations_as_list(ex1_full_cits)}\n\n"
        f"input 2: {ex2_text}{maybe_religion_line(cat_for_prompt, ex2_religion)}\n"
        f"output2: {format_full_citations_as_list(ex2_full_cits)}\n\n"
        f"input 3: {input3_text}{maybe_religion_line(cat_for_prompt, input3_religion)}\n"
        f"output3: "
    )

def encode_norm(texts: List[str]) -> torch.Tensor:
    return sbert.encode(texts, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)

def extract_query_text(item: Dict[str, Any]) -> str:
    if "query-text" in item and item["query-text"]:
        return item["query-text"]
    for k in ("input", "query", "text", "question"):
        if k in item and item[k]:
            return item[k]
    raise ValueError(f"Missing query text in item: {item}")

class StopOnStrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings):
        self.tokenizer = tokenizer
        self.stop_seqs = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]
    def __call__(self, input_ids, scores, **kwargs):
        ids = input_ids[0].tolist()
        for seq in self.stop_seqs:
            if len(seq) > 0 and ids[-len(seq):] == seq:
                return True
        return False

STOP_STRINGS = ["]", "\ninput", "\noutput"]
STOPPING = None if USE_OPENAI else StoppingCriteriaList([StopOnStrings(tokenizer, STOP_STRINGS)])

def extract_first_bracketed_list(text: str) -> str:
    m1 = re.search(r'\[', text)
    if not m1: return text.strip()
    start = m1.start()
    m2 = re.search(r'\]', text[start:])
    if not m2: return text[start:].strip()
    end = start + m2.end()
    return text[start:end].strip()

def generate_output(prompt: str, max_new: int = 512) -> str:
    if USE_OPENAI:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=max_new,
        )
        full_text = response.choices[0].message.content.strip()
        return extract_first_bracketed_list(full_text)
    else:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=TEMPERATURE,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=STOPPING,
            )
        full_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        marker = "output3:"
        after = full_text.split(marker, 1)[1] if marker in full_text else full_text
        return extract_first_bracketed_list(after)

print("[Data] Loading queries & examples …")
queries_data: List[Dict[str, Any]] = load_json(QUERIES_JSON_PATH)
examples_raw = load_json(EXAMPLES_JSON_PATH)
examples_data: List[Dict[str, Any]] = examples_raw if isinstance(examples_raw, list) else list(examples_raw.values())
print(f"[Data] Queries: {len(queries_data)} | Examples: {len(examples_data)}")

print("[Prep] Building category pools and embeddings …")
category_to_examples: Dict[str, List[Dict[str, Any]]] = {}
all_examples: List[Dict[str, Any]] = []
for ex in examples_data:
    if not ex.get("query-text") or not ex.get("full-citations"):
        continue
    all_examples.append(ex)
    cat = (ex.get("query-category") or "").strip()
    category_to_examples.setdefault(cat, []).append(ex)

all_texts = [ex["query-text"] for ex in all_examples]
all_embs = encode_norm(all_texts)
cat_embs: Dict[str, torch.Tensor] = {}
for cat, items in category_to_examples.items():
    texts = [it["query-text"] for it in items]
    if texts:
        cat_embs[cat] = encode_norm(texts)

def pick_two_examples(input_text: str, category: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    q = encode_norm([input_text])[0]
    if category in cat_embs and len(category_to_examples[category]) >= 2:
        pool = category_to_examples[category]
        embs = cat_embs[category]
    else:
        pool = all_examples
        embs = all_embs
    sims = util.cos_sim(q, embs)[0]
    k = min(2, sims.size(0))
    idxs = torch.topk(sims, k=k).indices.tolist()
    if len(idxs) == 1:
        return pool[idxs[0]], pool[idxs[0]]
    return pool[idxs[0]], pool[idxs[1]]

results = []
start_time = time.perf_counter()

print("[Run] Starting inference …")
pbar = tqdm(enumerate(queries_data), total=len(queries_data), desc="Infer", dynamic_ncols=True)
for idx, q in pbar:
    t0 = time.perf_counter()

    q_text = extract_query_text(q).strip()
    q_cat  = (q.get("query-category") or "").strip()
    q_rel  = (q.get("query-religion") or "Unknown").strip()

    ex1, ex2 = pick_two_examples(q_text, q_cat)

    prompt = build_two_shot_prompt(
        INSTRUCTION_BLOCK,
        cat_for_prompt=q_cat,
        ex1_text=ex1["query-text"], ex1_religion=ex1.get("query-religion"),
        ex1_full_cits=ex1["full-citations"],
        ex2_text=ex2["query-text"], ex2_religion=ex2.get("query-religion"),
        ex2_full_cits=ex2["full-citations"],
        input3_text=q_text, input3_religion=q_rel,
    )

    model_output = generate_output(prompt, max_new=512)

    out_item = {
        "id": q.get("id"),
        "query-text": q_text,
        "fewshot_example_ids": [ex1.get("id"), ex2.get("id")],
        "model_output": model_output,
    }
    if q_cat in CATS_REQUIRE_RELIGION:
        out_item["query-religion"] = q_rel

    results.append(out_item)

    if (idx + 1) % SAVE_EVERY == 0:
        write_json(OUTPUT_JSON_PATH, results)
        print(f"[Save] Wrote {len(results)} items to {OUTPUT_JSON_PATH} at idx {idx}")

    dt = time.perf_counter() - t0
    avg = (time.perf_counter() - start_time) / (idx + 1)
    pbar.set_postfix(elapsed=f"{dt:.2f}s", avg=f"{avg:.2f}s")

write_json(OUTPUT_JSON_PATH, results)
print(f"[Final] Wrote all {len(results)} items to {OUTPUT_JSON_PATH}")