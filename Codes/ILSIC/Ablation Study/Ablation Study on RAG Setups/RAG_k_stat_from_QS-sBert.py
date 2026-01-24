import os, json, re, torch, gc, traceback, numpy as np
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_ID             = "google/gemma-3-12b-it"  
MODEL_ID             = "meta-llama/Meta-Llama-3-8B-Instruct"  
PEFT_ADAPTER_PATH    = "/home/shounak/HDD/Layman-LSI/model/Gemma_3_LM_FT_bt_1_lr_1e5/checkpoint-4851"
FINETUNED_MODEL_PATH = None  # or full finetuned dir

TEST_DATA_PATH       = "/home/shounak/HDD/Layman-LSI/dataset/Layman-new-dataset/id-Layman-test-clean.json"

STATUTE_DEFS_PATH    = "/home/shounak/HDD/Layman-LSI/dataset/section-formal-texts.json"

ID_SECTION_META_PATH = "/home/shounak/HDD/Layman-LSI/dataset/id-section-meta.json"

OUTPUT_JSON_PATH     = "/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/RAG/QS-sbert/llama_base_RAG_k15_from_QS.json"
MISS_LOG_PATH        = "/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/RAG/QS-sbert/RAG_base_llama_defs_misses_k15_QS.json"
SKIP_LOG_PATH        = "/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/RAG/QS-sbert/RAG_base_llama_oom_skips_from_QS.json"

CACHE_DIR_LLMS       = "/home/shared/hf_cache"
CACHE_DIR_SBERT      = "/home/shared/hf_cache"

MAX_SEQ_LENGTH   = 32000
MAX_NEW_TOKENS   = 1024
TEMPERATURE      = 0.0
NUM_BEAMS        = 4
NO_REPEAT_NGRAM  = 8
BATCH_SAVE_N     = 10

SBERT_MODEL_ID       = "sentence-transformers/all-mpnet-base-v2"
MAX_CANDIDATES       = 15     # top-K statutes from definition similarity

TEST_QUERY_KEY       = "query-text"

USE_PEFT_ADAPTER = os.getenv("USE_PEFT_ADAPTER", "false").lower() in ("1", "true", "yes")

bits_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def load_model_and_tokenizer():
    print("[INFO] Loading tokenizer...")
    base_id_for_tok = FINETUNED_MODEL_PATH if FINETUNED_MODEL_PATH else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(base_id_for_tok, cache_dir=CACHE_DIR_LLMS)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    print("[INFO] Loading model...")

    max_memory = {
        0: "1GiB",   
        1: "45GiB",
        2: "45GiB",
        3: "45GiB",
    }
    base_id = FINETUNED_MODEL_PATH if FINETUNED_MODEL_PATH else MODEL_ID
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=bits_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        cache_dir=CACHE_DIR_LLMS
    )

   
    if not FINETUNED_MODEL_PATH and USE_PEFT_ADAPTER and PEFT_ADAPTER_PATH:
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT requested but `peft` is not installed.")
        print(f"[INFO] Attaching PEFT adapter from: {PEFT_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, PEFT_ADAPTER_PATH)

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    print("[INFO] Model and tokenizer loaded.")
    return model, tokenizer

def load_sbert() -> SentenceTransformer:
    if not SBERT_AVAILABLE:
        raise RuntimeError("Sentence-BERT not installed. `pip install sentence-transformers`")
    print(f"[INFO] Loading SBERT: {SBERT_MODEL_ID} on cuda:0")
    if torch.cuda.is_available():
        sbert = SentenceTransformer(SBERT_MODEL_ID, device="cuda:0", cache_folder=CACHE_DIR_SBERT)
        try:
            sbert = sbert.half()
        except Exception:
            print("[WARN] Could not switch SBERT to fp16; continuing in fp32.")
    else:
        print("[WARN] CUDA not available; SBERT on CPU.")
        sbert = SentenceTransformer(SBERT_MODEL_ID, device="cpu", cache_folder=CACHE_DIR_SBERT)

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

def build_full_to_id_map(id_section_meta: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Returns {"Section X of Act Y": "id"} when possible.
    """
    full2id: Dict[str, str] = {}
    for k, arr in id_section_meta.items():
        if not isinstance(arr, list) or len(arr) < 4:
            continue
        section = str(arr[0]).strip()
        act     = str(arr[1]).strip()
        cid     = str(arr[3]).strip()
        full    = f"{section} of {act}"
        if full and cid:
            full2id[full] = cid
    return full2id

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
                           candidates_full: List[str],
                           defs_map: Optional[Dict[str, str]],
                           defs_idx: Optional[Dict[str, str]],
                           miss_list_for_query: List[Any],
                           tokenizer: AutoTokenizer) -> str:
    query_block = f"QUERY:\n{whitespace_handler(query_text)}\n"
    cand_block  = "CANDIDATE_STATUTES:\n" + "\n".join([f'- \"{c}\"' for c in candidates_full])

    defs_block = ""
    if defs_map:
        defs_lines = []
        for c in candidates_full:
            if c in defs_map:
                d = whitespace_handler(str(defs_map[c]))
                defs_lines.append(f'- "{c}": {d}')
            else:
                dnorm = defs_idx.get(_norm_key(c)) if defs_idx else None
                if dnorm:
                    defs_lines.append(f'- "{c}": {whitespace_handler(str(dnorm))}')
                else:
                    miss_list_for_query.append({"candidate_full": c, "reason": "no exact or normalized match"})
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
    m = str(e).lower()
    return (
        isinstance(e, torch.cuda.OutOfMemoryError)
        or "out of memory" in m
        or "cuda error: out of memory" in m
        or ("cublas" in m and "alloc" in m)
    )

def precompute_def_embeddings(
    sbert: SentenceTransformer,
    defs_map: Dict[str, str],
) -> Tuple[List[str], np.ndarray]:
    """
    Build an array of embeddings for statute definitions.

    Returns:
      full_keys: list of "Section ... of Act ..." (same as defs_map keys order)
      def_embeddings: np.ndarray [N, D] normalized for cosine
    """
    full_keys = []
    texts     = []
    for full, d in defs_map.items():
        full_keys.append(full)
        texts.append(whitespace_handler(f"{full}. {d}"))

    print(f"[INFO] Encoding {len(texts)} statute definitions with SBERT...")
    emb = sbert.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    print("[INFO] Definition embeddings ready.")
    return full_keys, emb

def topk_by_defs(
    sbert: SentenceTransformer,
    defs_embeddings: np.ndarray,
    defs_full_keys: List[str],
    test_query_text: str,
    k: int = MAX_CANDIDATES,
) -> List[str]:
    """
    Encode test query, compute cosine with def embeddings, return top-k statute full strings.
    """
    q = whitespace_handler(test_query_text)
    q_emb = sbert.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    sims = np.dot(defs_embeddings, q_emb[0]) 
    idx_sorted = np.argsort(-sims)
    k = min(k, len(idx_sorted))
    picks = [defs_full_keys[i] for i in idx_sorted[:k]]

    seen, out = set(), []
    for p in picks:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

LLM_ROOT_DEVICE = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

def generate_selection(prompt: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> str:
    print("[DEBUG] Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                       max_length=tokenizer.model_max_length)

    print("[DEBUG] Moving tensors to LLM root device (cuda:1)...")
    inputs = {k: v.to(LLM_ROOT_DEVICE, non_blocking=True) for k, v in inputs.items()}

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

def main(model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    print(f"[INFO] Loading TEST from: {TEST_DATA_PATH}")
    test_rows = load_json(TEST_DATA_PATH)
    print(f"[INFO] Loaded {len(test_rows)} test queries.")

    print(f"[INFO] Loading statute DEFINITIONS from: {STATUTE_DEFS_PATH}")
    defs = load_json(STATUTE_DEFS_PATH)
    defs_idx = build_defs_index(defs) if defs else None
    if not defs or not isinstance(defs, dict):
        raise RuntimeError("STATUTE_DEFS_PATH must be a JSON object mapping 'Full Statute String' -> 'definition text'")
    print(f"[INFO] Loaded {len(defs)} statute definitions.")

    full2id = {}
    try:
        id_section_meta = load_json(ID_SECTION_META_PATH)
        full2id = build_full_to_id_map(id_section_meta)
        print(f"[INFO] Built full→id map for {len(full2id)} statutes.")
    except Exception:
        print("[WARN] Could not load ID_SECTION_META_PATH; continuing without ID mapping.")

    sbert = load_sbert()
    defs_full_keys, defs_embeds = precompute_def_embeddings(sbert, defs)

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    results: List[Dict[str, Any]] = []
    miss_log: List[Dict[str, Any]] = []
    skips: List[Dict[str, Any]] = []

    for i, row in enumerate(tqdm(test_rows, desc="SBERT(query→defs)→topK→LLM selection"), start=1):
        qid = row.get("id")
        print(f"\n[INFO] Processing test id={qid} ...")

        query_text = whitespace_handler(row.get(TEST_QUERY_KEY, ""))

        print("[DEBUG] Retrieving top-K statutes from definition embeddings...")
        candidates_full = topk_by_defs(
            sbert=sbert,
            defs_embeddings=defs_embeds,
            defs_full_keys=defs_full_keys,
            test_query_text=query_text,
            k=MAX_CANDIDATES
        )
        print(f"[DEBUG] {len(candidates_full)} candidate statute(s).")

        misses_this: List[Any] = []

        print("[DEBUG] Building prompt...")
        prompt = build_prompt_for_model(
            MODEL_ID, query_text, candidates_full, defs, defs_idx, misses_this, tokenizer
        )
        try:
            raw = generate_selection(prompt, tokenizer, model)
            picked = parse_bracketed_list(raw)
            picked_filtered = [p for p in picked if p in candidates_full]

            picked_ids = [full2id.get(p, None) for p in picked_filtered]
            candidate_ids = [full2id.get(c, None) for c in candidates_full]

            print(f"[DEBUG] Selected {len(picked_filtered)} statute(s).")

            results.append({
                "id": qid,
                "input": query_text,
                "candidate_statutes": candidates_full,
                "candidate_ids": candidate_ids,         
                "selected_statutes": picked_filtered,
                "selected_ids": picked_ids,              
                "raw_generation": raw
            })

            if misses_this:
                miss_log.append({
                    "id": qid,
                    "missing": misses_this
                })

        except Exception as e:
            reason = "OOM" if _is_oom_error(e) else "OtherError"
            tb_tail = "\n".join(traceback.format_exc().splitlines()[-5:])
            print(f"[ERROR] {reason} on id={qid}. Skipping. Details:\n{tb_tail}")

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            gc.collect()

            skips.append({
                "id": qid,
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
    torch.set_grad_enabled(False)  
    model, tokenizer = load_model_and_tokenizer()
    main(model, tokenizer)
    print("[INFO] Done.")
