#!/usr/bin/env python3
import os, json, re, torch, gc, traceback
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

MODEL_ID             = "google/gemma-3-12b-it"  # base model
PEFT_ADAPTER_PATH    = "/home/shounak/HDD/Layman-LSI/model/Gemma_3_LM_FT_bt_1_lr_1e5/checkpoint-4851"  
FINETUNED_MODEL_PATH = None  

MERGED_INF_PATH      = "/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/data/inf_data_merged.json"
STATUTE_DEFS_PATH    = "/home/shounak/HDD/Layman-LSI/dataset/section-formal-texts.json"  
OUTPUT_JSON_PATH     = "/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/RAG/gemma_base_RAG_inf.json"
MISS_LOG_PATH        = "/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/RAG/RAG_base_section_def_misses.json"
SKIP_LOG_PATH        = "/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/RAG/RAG_base_oom_skips.json"


# Toggle this to choose whether to attach the PEFT adapter to the base model.
# True  -> use base + adapter (if PEFT_ADAPTER_PATH is set)
# False -> use base model ONLY
USE_PEFT_ADAPTER = os.getenv("USE_PEFT_ADAPTER", "false").lower() in ("1", "true", "yes")


MAX_SEQ_LENGTH   = 32000
MAX_NEW_TOKENS   = 1024
TEMPERATURE      = 0.0
NUM_BEAMS        = 4
NO_REPEAT_NGRAM  = 8
BATCH_SAVE_N     = 10   # save outputs + miss/skip logs every 10 queries

bits_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer():
    print("[INFO] Loading tokenizer...")
    base_id_for_tok = FINETUNED_MODEL_PATH if FINETUNED_MODEL_PATH else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(base_id_for_tok)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    print("[INFO] Loading model...")
    if FINETUNED_MODEL_PATH:
        model = AutoModelForCausalLM.from_pretrained(
            FINETUNED_MODEL_PATH,
            quantization_config=bits_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="/home/shared/hf_cache"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bits_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="/home/shared/hf_cache"
        )
        if USE_PEFT_ADAPTER and PEFT_ADAPTER_PATH:
            if not PEFT_AVAILABLE:
                raise RuntimeError("PEFT adapter path provided, but `peft` is not installed.")
            print(f"[INFO] Attaching PEFT adapter from: {PEFT_ADAPTER_PATH}")
            model = PeftModel.from_pretrained(model, PEFT_ADAPTER_PATH)
        else:
            if not USE_PEFT_ADAPTER:
                print("[INFO] USE_PEFT_ADAPTER=False -> using BASE MODEL only (no adapter).")
            elif not PEFT_ADAPTER_PATH:
                print("[INFO] No PEFT_ADAPTER_PATH set -> using BASE MODEL only.")

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    print("[INFO] Model and tokenizer loaded successfully.")
    return model, tokenizer

def whitespace_handler(text: str) -> str:
    text = re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', text.strip()))
    text = text.replace("\xad", "")
    return text

def parse_bracketed_list(text: str) -> List[str]:
    """
    Extract items like ["Section X of Act"; "Section Y of Act"] or comma-separated variants.
    """
    m = re.search(r'\[(.*)\]', text, flags=re.DOTALL)
    if not m:
        return []
    inner = m.group(1)
    items = re.findall(r'"([^"]+)"', inner)
    seen, out = set(), []
    for it in items:
        it = it.strip()
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _norm_key(s: str) -> str:
    """
    Light normalization to improve matching between candidate statute strings and defs keys.
    - lowercase
    - collapse spaces
    - drop leading/trailing spaces
    - remove punctuation (keep letters, digits, spaces)
    - also normalize ' of the ' -> ' of ' to ignore optional 'The'
    """
    s = s.lower()
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace(" of the ", " of ")  
    s = re.sub(r'[^a-z0-9 ]+', '', s)  
    return s

def build_defs_index(defs_map: Dict[str, str]) -> Dict[str, str]:
    """Create a normalized-key index for the defs map."""
    return { _norm_key(k): v for k, v in defs_map.items() }


SYSTEM_RULES = """
You are a legal expert on Indian law. Select statutes that best apply to the user's situation, but ONLY from the provided CANDIDATE_STATUTES.

Rules:
1) Output exactly one list in square brackets using double quotes, e.g. ["Section X of Act"; "Section Y of Act"].
2) Each entry must be a quoted string in the exact same form as provided in CANDIDATE_STATUTES (no rephrasing).
3) Do not add anything outside that one list. No commentary.
4) Do not invent or add statutes not in CANDIDATE_STATUTES.
5) Prefer fewer, more precise provisions over many generic ones.
6) If none clearly apply, return [].
"""

def build_prompt_for_model(model_name: str,
                           query_text: str,
                           candidates: List[str],
                           defs_map: Optional[Dict[str, str]],
                           defs_idx: Optional[Dict[str, str]],
                           miss_list_for_query: List[Any]) -> str:
    """
    Build prompt with definitions.
    Matching strategy:
      1) exact key in defs_map
      2) else normalized key in defs_idx (built from defs_map)
      3) else log miss
    """
    query_block = f"QUERY:\n{whitespace_handler(query_text)}\n"
    cand_block = "CANDIDATE_STATUTES:\n" + "\n".join([f'- "{c}"' for c in candidates])

    defs_block = ""
    if defs_map:
        defs_lines = []
        for c in candidates:
            if c in defs_map:
                d = whitespace_handler(str(defs_map[c]))
                defs_lines.append(f'- "{c}": {d}')
            else:
                dnorm = defs_idx.get(_norm_key(c)) if defs_idx else None
                if dnorm:
                    dnorm = whitespace_handler(str(dnorm))
                    defs_lines.append(f'- "{c}": {dnorm}')
                else:
                    miss_list_for_query.append({"candidate": c, "reason": "no exact or normalized match"})
                    print(f"[WARN] No definition found for candidate (after normalization): {c}")
        if defs_lines:
            defs_block = "STATUTE_DEFINITIONS:\n" + "\n".join(defs_lines)

    if "llama" in model_name.lower():
        messages = [
            {"role": "system", "content": SYSTEM_RULES},
            {"role": "user", "content": f"{query_block}\n{cand_block}\n\n{defs_block}\n\nReturn only the bracketed list."}
        ]
    elif "gemma" in model_name.lower():
        user_msg = f"{SYSTEM_RULES}\n\n{query_block}\n{cand_block}\n\n{defs_block}\n\nReturn only the bracketed list."
        messages = [{"role": "user", "content": user_msg}]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_RULES},
            {"role": "user", "content": f"{query_block}\n{cand_block}\n\n{defs_block}\n\nReturn only the bracketed list."}
        ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def _is_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        isinstance(e, torch.cuda.OutOfMemoryError)
        or "out of memory" in msg
        or "cuda error: out of memory" in msg
        or ("cublas" in msg and "alloc" in msg)
    )

def generate_selection(prompt: str) -> str:
    """
    Returns decoded string on success.
    Raises Exception (including OOM) on failure; caller decides how to handle/skip.
    """
    print("[DEBUG] Tokenizing prompt...")
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length
    )

    print("[DEBUG] Moving tensors to device...")
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    print("[DEBUG] Generating output...")
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=TEMPERATURE,
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("[DEBUG] Decoding output...")
    gen = outputs[0][input_len:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def main():
    print(f"[INFO] Loading merged inference data from: {MERGED_INF_PATH}")
    merged = load_json(MERGED_INF_PATH)
    print(f"[INFO] Loaded {len(merged)} queries.")

    defs = load_json(STATUTE_DEFS_PATH) if STATUTE_DEFS_PATH else None
    defs_idx = build_defs_index(defs) if defs else None

    if defs:
        print(f"[INFO] Loaded {len(defs)} statute definitions. (normalized index ready)")

    results: List[Dict[str, Any]] = []
    miss_log: List[Dict[str, Any]] = []
    skips: List[Dict[str, Any]] = []

    for i, row in enumerate(tqdm(merged, desc="Selecting statutes"), start=1):
        qn = row.get("query_no")
        print(f"\n[INFO] Processing query {qn}...")

        query_text = row.get("input", "")
        candidates_raw = (row.get("model_output", []) or [])

      
        seen, candidates = set(), []
        for c in candidates_raw:
            cs = c.strip()
            if cs and cs not in seen:
                seen.add(cs)
                candidates.append(cs)

        print(f"[DEBUG] {len(candidates)} candidate statute(s) found.")

        misses_this_query: List[Any] = []

        print("[DEBUG] Building prompt...")
        prompt = build_prompt_for_model(MODEL_ID, query_text, candidates, defs, defs_idx, misses_this_query)

        try:
            raw = generate_selection(prompt)
            picked = parse_bracketed_list(raw)
            picked_filtered = [p for p in picked if p in candidates]  # strict subset

            print(f"[DEBUG] Selected {len(picked_filtered)} statute(s).")

            results.append({
                "query_no": qn,
                "input": query_text,
                "candidate_statutes": candidates,
                "selected_statutes": picked_filtered,
                "raw_generation": raw
            })

            if misses_this_query:
                miss_log.append({
                    "query_no": qn,
                    "missing_definitions": misses_this_query
                })
                print(f"[WARN] Missing definitions for {len(misses_this_query)} statute(s).")

        except Exception as e:
            reason = "OOM" if _is_oom_error(e) else "OtherError"
            tb_tail = "\n".join(traceback.format_exc().splitlines()[-5:])
            print(f"[ERROR] {reason} on query {qn}. Skipping. Details:\n{tb_tail}")

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            gc.collect()

            skips.append({
                "query_no": qn,
                "reason": reason,
                "error": str(e),
            })

        if i % BATCH_SAVE_N == 0:
            print(f"[INFO] Auto-saving after {i} queries...")
            save_json(OUTPUT_JSON_PATH, results)
            save_json(MISS_LOG_PATH, miss_log)
            save_json(SKIP_LOG_PATH, skips)

    print("[INFO] Final save...")
    save_json(OUTPUT_JSON_PATH, results)
    save_json(MISS_LOG_PATH, miss_log)
    save_json(SKIP_LOG_PATH, skips)
    print(f"[OK] Saved {len(results)} results to {OUTPUT_JSON_PATH}")
    print(f"[OK] Saved {len(miss_log)} miss-log entries to {MISS_LOG_PATH}")
    print(f"[OK] Saved {len(skips)} skipped queries to {SKIP_LOG_PATH}")


if __name__ == "__main__":
    print("[INFO] Starting pipeline...")
    model, tokenizer = load_model_and_tokenizer()
    main()
    print("[INFO] Done.")
