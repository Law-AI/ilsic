"""
===========================================================
 GPU-ISOLATED + ID-RESUME ENABLED RAG SCRIPT
===========================================================

This updated script includes:

1. GPU Isolation (Option X)
   - LLM uses GPUs declared in LLM_GPU_VISIBLE
   - SBERT uses SBERT_GPU_DEVICE exclusively

2. Resume Logic Upgrade
   - Only process: IDs present in TEST but NOT in OUTPUT
   - `--rerun-skipped` only processes true skipped entries
   - `--force-all` processes entire TEST from scratch

3. Code reorganized for clarity
4. No other logic changed
===========================================================
"""

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
LLM_GPU_VISIBLE = "3,2"     
SBERT_GPU_DEVICE = "cuda:1"   # ← SBERT will run ONLY on this GPU (or "cpu")

import torch

os.environ["CUDA_VISIBLE_DEVICES"] ="LLM_GPU_VISIBLE"


device = torch.device("cuda") 

import json, re, torch, gc, traceback, numpy as np, argparse
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

MODEL_ID             = "google/gemma-3-12b-it"
# MODEL_ID             = "meta-llama/Meta-Llama-3-8B-Instruct"
PEFT_ADAPTER_PATH    = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/model/Llama_3_LM_FT_bt_1_lr_1e5/checkpoint-4851"
FINETUNED_MODEL_PATH = None

TEST_DATA_PATH       = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/Layman-new-dataset/id-Layman-test-clean-new.json"
TRAIN_DATA_PATH      = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/Layman-new-dataset/id-Layman-train-clean-new.json"

STATUTE_DEFS_PATH    = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/section-formal-texts.json"
ID_SECTION_META_PATH = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/id-section-meta.json"

OUTPUT_JSON_PATH     = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/Rebuttal_ARR_Oct_2025/FT-model-on-RAG/gemma_FT_RAG_kStatfrom_10-CID_QQ_Restrict.json"
MISS_LOG_PATH        = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/Rebuttal_ARR_Oct_2025/FT-model-on-RAG/RAG_gemma_section_def_misses__kStatfrom_10-CID_QQ_Restrict.json"
SKIP_LOG_PATH        = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/Rebuttal_ARR_Oct_2025/FT-model-on-RAG/RAG_gemma_oom_skips__kStatfrom_QQ_10-CID_Restrict.json"

CACHE_DIR_LLMS       = "/home/shared/hf_cache"
CACHE_DIR_SBERT      = "/home/shared/hf_cache"

USE_PEFT_ADAPTER = True

MAX_SEQ_LENGTH   = 20000
MAX_NEW_TOKENS   = 2048
TEMPERATURE      = 0.0
NUM_BEAMS        = 4
NO_REPEAT_NGRAM  = 8
BATCH_SAVE_N     = 10

SBERT_MODEL_ID       = "sentence-transformers/all-mpnet-base-v2"
MAX_CANDIDATES       = 10
TOPK_SIM_TRAIN       = 500
SBERT_DEVICE         = SBERT_GPU_DEVICE   
TEST_QUERY_KEY       = "query-text"
TRAIN_QUERY_KEY      = "query-text"
TRAIN_ID_KEYS        = ["citations-id"]

bits_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"[INFO] LLM running on {device}, GPUs visible = {LLM_GPU_VISIBLE}")

def load_model_and_tokenizer():
    print("[INFO] Loading tokenizer...")
    base_tok_id = FINETUNED_MODEL_PATH if FINETUNED_MODEL_PATH else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(
        base_tok_id,
        cache_dir=CACHE_DIR_LLMS
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    print("[INFO] Loading LLM model...")
    if FINETUNED_MODEL_PATH:
        model = AutoModelForCausalLM.from_pretrained(
            FINETUNED_MODEL_PATH,
            quantization_config=bits_config,
            torch_dtype=torch.bfloat16,
            device_map={"": 3},
            cache_dir=CACHE_DIR_LLMS
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bits_config,
            torch_dtype=torch.bfloat16,
            device_map={"": 3},
            cache_dir=CACHE_DIR_LLMS
        )

        if USE_PEFT_ADAPTER and PEFT_ADAPTER_PATH:
            if not PEFT_AVAILABLE:
                raise RuntimeError("PEFT adapter required but peft not installed.")
            print(f"[INFO] Attaching PEFT adapter: {PEFT_ADAPTER_PATH}")
            model = PeftModel.from_pretrained(model, PEFT_ADAPTER_PATH,device_map={"": 3}
)

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    print("[INFO] LLM loaded successfully.")
    return model, tokenizer

def load_sbert() -> SentenceTransformer:
    if not SBERT_AVAILABLE:
        raise RuntimeError(
            "Sentence-BERT not installed. 'pip install sentence-transformers'"
        )

    print(f"[INFO] Loading SBERT on: {SBERT_DEVICE}")
    sbert = SentenceTransformer(
        SBERT_MODEL_ID,
        device=SBERT_DEVICE,
        cache_folder=CACHE_DIR_SBERT
    )
    sbert.eval()
    return sbert

def whitespace_handler(text: str) -> str:
    text = re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', str(text).strip()))
    text = text.replace("\xad", "")
    return text

def parse_bracketed_list(text: str) -> List[str]:
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
    s = s.lower()
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace(" of the ", " of ")
    s = re.sub(r'[^a-z0-9 ]+', '', s)
    return s

def build_defs_index(defs_map: Dict[str, str]) -> Dict[str, str]:
    return { _norm_key(k): v for k, v in defs_map.items() }

def extract_query_text(row: Dict[str, Any], key: str) -> str:
    return whitespace_handler(row.get(key, ""))

def coerce_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [whitespace_handler(i) for i in x if str(i).strip()]
    return [whitespace_handler(x)]

def extract_id_list(row: Dict[str, Any], id_keys: List[str]) -> List[str]:
    seen, out = set(), []
    for k in id_keys:
        vals = coerce_list(row.get(k, []))
        for v in vals:
            v = str(v).strip()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
    return out

SYSTEM_RULES = """
You are a legal expert on Indian law. Select statutes that best apply to the user’s situation, but ONLY from the provided CANDIDATE_STATUTES.
Rules:
1) Output exactly one list in square brackets using double quotes.
2) Each entry must match EXACTLY the provided statute strings.
3) No commentary.
4) Do not invent new statutes.
5) Prefer fewer but more precise statutes.
6) If none apply, return [].
"""

def build_prompt_for_model(
    model_name: str,
    query_text: str,
    candidates_full: List[str],
    defs_map: Optional[Dict[str, str]],
    defs_idx: Optional[Dict[str, str]],
    miss_list_for_query: List[Any],
    tokenizer: AutoTokenizer
) -> str:

    query_block = f"QUERY:\n{whitespace_handler(query_text)}\n"
    cand_block  = "CANDIDATE_STATUTES:\n" + "\n".join([f'- "{c}"' for c in candidates_full])

    defs_block = ""
    if defs_map:
        lines = []
        for c in candidates_full:
            if c in defs_map:
                desc = whitespace_handler(str(defs_map[c]))
                lines.append(f'- "{c}": {desc}')
            else:
                dnorm = defs_idx.get(_norm_key(c)) if defs_idx else None
                if dnorm:
                    lines.append(f'- "{c}": {whitespace_handler(str(dnorm))}')
                else:
                    miss_list_for_query.append({"candidate_full": c, "reason": "no match in definitions"})
        if lines:
            defs_block = "STATUTE_DEFINITIONS:\n" + "\n".join(lines)

    if "gemma" in model_name.lower():
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
    m = str(e).lower()
    return (
        isinstance(e, torch.cuda.OutOfMemoryError)
        or "out of memory" in m
        or "cuda error: out of memory" in m
        or ("cublas" in m and "alloc" in m)
    )

def precompute_train_sbert(
    sbert: SentenceTransformer,
    train_rows: List[Dict[str, Any]],
) -> Tuple[List[str], np.ndarray, List[List[str]]]:

    print("[INFO] Preparing train query texts + IDs...")

    tq_texts, tq_ids = [], []
    for row in train_rows:
        text = extract_query_text(row, TRAIN_QUERY_KEY)
        ids  = extract_id_list(row, TRAIN_ID_KEYS)
        tq_texts.append(text)
        tq_ids.append(ids)

    print(f"[INFO] Encoding SBERT embeddings for {len(tq_texts)} train queries...")

    embeds = sbert.encode(
        tq_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    print("[INFO] SBERT train embeddings ready.")
    return tq_texts, embeds, tq_ids


def build_candidates_ids_from_sbert(
    sbert: SentenceTransformer,
    train_embeddings: np.ndarray,
    train_ids_list: List[List[str]],
    test_query_text: str,
    max_candidates: int = MAX_CANDIDATES,
    topk_sim_train: int = TOPK_SIM_TRAIN,
) -> List[str]:

    q = whitespace_handler(test_query_text)
    q_emb = sbert.encode([q], convert_to_numpy=True, normalize_embeddings=True)

    sims = np.dot(train_embeddings, q_emb[0])
    idx_sorted = np.argsort(-sims)

    seen, out = set(), []
    limit = min(topk_sim_train, len(idx_sorted))

    for i in range(limit):
        ri = idx_sorted[i]
        for cid in train_ids_list[ri]:
            cid = str(cid).strip()
            if cid and cid not in seen:
                seen.add(cid)
                out.append(cid)
                if len(out) >= max_candidates:
                    return out
    return out

def build_id_to_full_map(id_section_meta: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
    id2full = {}
    for k, arr in id_section_meta.items():
        if not isinstance(arr, list) or len(arr) < 4:
            continue

        section = arr[0]
        act     = arr[1]
        blurb   = arr[2]
        cid     = str(arr[3]).strip()

        if cid:
            full = f"{section} of {act}"
            id2full[cid] = {
                "full": full,
                "section": section,
                "act": act,
                "blurb": blurb,
                "key": k,
            }
    return id2full


def ids_to_full_citations(
    cand_ids: List[str],
    id2full: Dict[str, Dict[str, str]]
) -> Tuple[List[str], List[Dict[str, str]]]:

    out, misses, seen = [], [], set()
    for cid in cand_ids:
        meta = id2full.get(str(cid).strip())
        if meta:
            full = meta["full"]
            if full and full not in seen:
                seen.add(full)
                out.append(full)
        else:
            misses.append({"id": cid, "reason": "id not found"})
    return out, misses

def generate_selection(prompt: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> str:
    print("[DEBUG] Tokenizing prompt...")
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length
    )
    print("[DEBUG] Moving to device...")
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

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    gen = outputs[0][input_len:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def main(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args):

    print(f"[INFO] Loading TEST: {TEST_DATA_PATH}")
    test_rows = load_json(TEST_DATA_PATH)
    print(f"[INFO] Loaded {len(test_rows)} test rows.")

    results, miss_log, skips = [], [], []
    if os.path.exists(OUTPUT_JSON_PATH):
        results = load_json(OUTPUT_JSON_PATH)
    if os.path.exists(MISS_LOG_PATH):
        miss_log = load_json(MISS_LOG_PATH)
    if os.path.exists(SKIP_LOG_PATH):
        skips = load_json(SKIP_LOG_PATH)

    test_ids    = {row.get("id") for row in test_rows if row.get("id") is not None}
    output_ids  = {r.get("id")   for r in results  if r.get("id") is not None}
    skipped_ids = {s.get("id")   for s in skips    if s.get("id") is not None}

    remaining_ids = test_ids - output_ids

    print(f"[INFO] Total test IDs      : {len(test_ids)}")
    print(f"[INFO] IDs already output  : {len(output_ids)}")
    print(f"[INFO] Remaining to process: {len(remaining_ids)}")
    print(f"[INFO] Skipped earlier     : {len(skipped_ids)}")

    if args.force_all:
        print("[RESUME] --force-all: processing ALL test queries.")
        to_process = test_rows

    elif args.rerun_skipped:
        print("[RESUME] --rerun-skipped: processing only skipped IDs.")
        to_process = [r for r in test_rows if r.get("id") in skipped_ids]

    else:
        print("[RESUME] Default resume: processing remaining IDs only.")
        to_process = [r for r in test_rows if r.get("id") in remaining_ids]

    print(f"[RESUME] Will process: {len(to_process)} queries.")

    print(f"[INFO] Loading TRAIN: {TRAIN_DATA_PATH}")
    train_rows = load_json(TRAIN_DATA_PATH)

    print(f"[INFO] Loading ID→SECTION meta...")
    id_section_meta = load_json(ID_SECTION_META_PATH)
    id2full = build_id_to_full_map(id_section_meta)

    defs = load_json(STATUTE_DEFS_PATH) if STATUTE_DEFS_PATH else None
    defs_idx = build_defs_index(defs) if defs else None

    sbert = load_sbert()
    _, train_embeds, train_ids_list = precompute_train_sbert(sbert, train_rows)

    processed_since_save = 0
    for row in tqdm(to_process, desc="Processing queries..."):

        qid = row.get("id")
        print(f"\n[INFO] --- Processing test ID={qid} ---")

        query_text = extract_query_text(row, TEST_QUERY_KEY)
        misses_this = []

        try:
            print("[DEBUG] Building SBERT candidates...")
            candidate_ids = build_candidates_ids_from_sbert(
                sbert,
                train_embeds,
                train_ids_list,
                query_text,
                MAX_CANDIDATES,
                TOPK_SIM_TRAIN,
            )
            print(f"[DEBUG] Candidate IDs = {candidate_ids}")

        except Exception as e:
            reason = "SBERTError"
            tb = "\n".join(traceback.format_exc().splitlines()[-5:])
            print(f"[ERROR] SBERT failure for ID={qid}:\n{tb}")

            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

            gc.collect()

            skips.append({"id": qid, "reason": reason, "error": str(e)})

            processed_since_save += 1
            if processed_since_save >= BATCH_SAVE_N:
                save_json(OUTPUT_JSON_PATH, results)
                save_json(MISS_LOG_PATH, miss_log)
                save_json(SKIP_LOG_PATH, skips)
                processed_since_save = 0

            continue

        candidates_full, id_misses = ids_to_full_citations(candidate_ids, id2full)
        if id_misses:
            misses_this.append({"id_mapping_misses": id_misses})

        print("[DEBUG] Building prompt...")
        prompt = build_prompt_for_model(
            MODEL_ID,
            query_text,
            candidates_full,
            defs,
            defs_idx,
            misses_this,
            tokenizer
        )

        try:
            raw = generate_selection(prompt, tokenizer, model)
            picked = parse_bracketed_list(raw)
            picked_filtered = [p for p in picked if p in candidates_full]

            print(f"[DEBUG] Selected {len(picked_filtered)} statutes.")

            results.append({
                "id": qid,
                "input": query_text,
                "candidate_ids": candidate_ids,
                "candidate_statutes": candidates_full,
                "selected_statutes": picked_filtered,
                "raw_generation": raw
            })

            if misses_this:
                miss_log.append({"id": qid, "missing": misses_this})

        except Exception as e:
            reason = "OOM" if _is_oom_error(e) else "OtherError"
            tb = "\n".join(traceback.format_exc().splitlines()[-5:])
            print(f"[ERROR] Generation failure for ID={qid}:\n{tb}")

            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

            gc.collect()

            skips.append({"id": qid, "reason": reason, "error": str(e)})

        processed_since_save += 1

        if processed_since_save >= BATCH_SAVE_N:
            print("[INFO] Auto-saving...")
            save_json(OUTPUT_JSON_PATH, results)
            save_json(MISS_LOG_PATH, miss_log)
            save_json(SKIP_LOG_PATH, skips)
            processed_since_save = 0

    print("[INFO] Final saving of all results/logs...")
    save_json(OUTPUT_JSON_PATH, results)
    save_json(MISS_LOG_PATH, miss_log)
    save_json(SKIP_LOG_PATH, skips)

    print(f"[OK] Results saved: {len(results)}")
    print(f"[OK] Miss log saved: {len(miss_log)}")
    print(f"[OK] Skips saved  : {len(skips)}")

def build_argparser():
    ap = argparse.ArgumentParser(description="GPU-isolated SBERT→RAG pipeline")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Resume from output/miss/skip files (default).")
    ap.add_argument("--rerun-skipped", action="store_true",
                    help="Re-run only the skipped queries.")
    ap.add_argument("--force-all", action="store_true",
                    help="Process ALL test queries from scratch.")

    return ap

if __name__ == "__main__":
    print("[INFO] Starting pipeline...")
    args = build_argparser().parse_args()
    model, tokenizer = load_model_and_tokenizer()
    main(model, tokenizer, args)
    print("[INFO] Done.")
