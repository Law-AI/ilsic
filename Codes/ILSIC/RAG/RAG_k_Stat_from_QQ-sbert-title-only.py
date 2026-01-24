"""
===========================================================
 HOW TO RUN THIS SCRIPT (with resume options)
===========================================================

Default (resume from last saved query):
    python ./dev/RAG_k_stat_from_QQ-sBert.py

Resume from last saved results (same as default):
    python ./dev/RAG_k_stat_from_QQ-sBert.py --resume

Only re-run previously skipped queries (from SKIP_LOG_PATH):
    python ./dev/RAG_k_stat_from_QQ-sBert.py --rerun-skipped

Force fresh run from scratch (ignore saved results/logs):
    python ./dev/RAG_k_stat_from_QQ-sBert.py --force-all

Notes:
- Script auto-saves every BATCH_SAVE_N queries.
- OUTPUT_JSON_PATH, MISS_LOG_PATH, SKIP_LOG_PATH are updated incrementally.
- Resume uses the query "id" field to skip already processed items.
- You can set SBERT_DEVICE="cpu" or "cuda:2" in env vars if you want SBERT isolated.

===========================================================
"""

import os, json, re, torch, gc, traceback, numpy as np, argparse
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
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"

# MODEL_ID             = "google/gemma-3-12b-it"
MODEL_ID             = "meta-llama/Meta-Llama-3-8B-Instruct"
PEFT_ADAPTER_PATH    = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/model/Gemma_3_LM_FT_bt_1_lr_1e5/checkpoint-4851"
FINETUNED_MODEL_PATH = None

TEST_DATA_PATH       = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/Layman-new-dataset/id-Layman-test-clean-new.json"
TRAIN_DATA_PATH      = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/Layman-new-dataset/id-Layman-train-clean-new.json"

STATUTE_DEFS_PATH    = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/section-formal-texts.json"
ID_SECTION_META_PATH = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/id-section-meta.json"

OUTPUT_JSON_PATH     = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/Rebuttal_ARR_Oct_2025/restricted/title-only/llama_base_RAG_kStatfrom_title-only_QQ_noRestrict.json"
MISS_LOG_PATH        = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/Rebuttal_ARR_Oct_2025/restricted/title-only/RAG_llama_section_def_misses_kStatfrom_title-only_QQ_Restrict.json"
SKIP_LOG_PATH        = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/Rebuttal_ARR_Oct_2025/restricted/title-only/RAG_llama_oom_skips_kStatfrom_title-only_QQ_Restrict.json"

CACHE_DIR_LLMS       = "/home/shared/hf_cache"
CACHE_DIR_SBERT      = "/home/shared/hf_cache"

USE_PEFT_ADAPTER = os.getenv("USE_PEFT_ADAPTER", "false").lower() in ("1", "true", "yes")

MAX_SEQ_LENGTH   = 32000
MAX_NEW_TOKENS   = 8124
TEMPERATURE      = 0.0
NUM_BEAMS        = 4
NO_REPEAT_NGRAM  = 8
BATCH_SAVE_N     = 10

SBERT_MODEL_ID       = "sentence-transformers/all-mpnet-base-v2"
MAX_CANDIDATES       = 15
TOPK_SIM_TRAIN       = 500
SBERT_DEVICE         = os.getenv("SBERT_DEVICE", "cuda:0")

TITLE_MATCH_THRESHOLD = 70  

TEST_QUERY_KEY       = "query-text"
TRAIN_QUERY_KEY      = "query-text"
TRAIN_ID_KEYS        = ["citations-id"]

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
    tokenizer = AutoTokenizer.from_pretrained(base_id_for_tok, cache_dir=CACHE_DIR_LLMS)
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
            cache_dir=CACHE_DIR_LLMS
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bits_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=CACHE_DIR_LLMS
        )
        if USE_PEFT_ADAPTER and PEFT_ADAPTER_PATH:
            if not PEFT_AVAILABLE:
                raise RuntimeError("PEFT requested but `peft` is not installed.")
            print(f"[INFO] Attaching PEFT adapter from: {PEFT_ADAPTER_PATH}")
            model = PeftModel.from_pretrained(model, PEFT_ADAPTER_PATH)
        else:
            if not USE_PEFT_ADAPTER:
                print("[INFO] Using BASE MODEL only (no adapter).")
            elif not PEFT_ADAPTER_PATH:
                print("[INFO] No PEFT_ADAPTER_PATH found; using base model only.")

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    print("[INFO] Model and tokenizer loaded.")
    return model, tokenizer


def load_sbert() -> SentenceTransformer:
    if not SBERT_AVAILABLE:
        raise RuntimeError("Sentence-BERT not installed. `pip install sentence-transformers`")
    print(f"[INFO] Loading SBERT: {SBERT_MODEL_ID} on {SBERT_DEVICE}")
    sbert = SentenceTransformer(SBERT_MODEL_ID, device=SBERT_DEVICE, cache_folder=CACHE_DIR_SBERT)
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

CONNECTIVE_FILLERS = {
    "under", "of", "by", "on", "in", "at", "with", "and", "for", "from", "to",
    "a", "an", "or", "but", "as", "per", "via", "upon", "within", "into",
    "through", "over", "between", "among", "around", "about", "before", "after",
    "during", "since", "until", "towards", "against", "without", "including",
    "regarding", "concerning", "according", "versus", "vs", "w.r.t",
}

def remove_connective_fillers(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    words = text.split()
    filtered = []
    for word in words:
        clean_word = "".join(c for c in word if c.isalnum())
        if clean_word not in CONNECTIVE_FILLERS or any(c.isdigit() for c in clean_word):
            filtered.append(clean_word)
    return " ".join(filtered).strip()


def normalize_for_comparison(text: str) -> str:
    if not text:
        return ""
    text = remove_connective_fillers(text)
    words = text.split()
    filtered = []

    for word in words:
        clean_word = "".join(c for c in word if c.isalnum())
        if (
            any(c.isdigit() for c in clean_word)
            or len(clean_word) > 2
            or (len(clean_word) == 2 and clean_word.isalpha())
        ):
            filtered.append(clean_word)

    if len(filtered) < len(words) / 3:
        return " ".join("".join(c for c in w if c.isalnum()) for w in words if w)

    return " ".join(filtered)

from rapidfuzz import fuzz

def smart_fuzzy_match(text1: str, text2: str) -> float:
    """Hybrid fuzzy scoring used for section-act matching."""
    if not text1 or not text2:
        return 0.0

    clean1 = remove_connective_fillers(text1)
    clean2 = remove_connective_fillers(text2)

    norm1 = normalize_for_comparison(clean1)
    norm2 = normalize_for_comparison(clean2)

    if norm1 == norm2:
        return 100.0

    scores = [
        fuzz.ratio(clean1, clean2),
        fuzz.token_sort_ratio(clean1, clean2),
        fuzz.token_set_ratio(clean1, clean2),
        fuzz.ratio(norm1, norm2),
        fuzz.token_sort_ratio(norm1, norm2),
        fuzz.token_set_ratio(norm1, norm2),
    ]

    w1, w2 = set(clean1.split()), set(clean2.split())
    if w1 and w2:
        inter = len(w1 & w2)
        uni = len(w1 | w2)
        if uni > 0:
            scores.append((inter / uni) * 100)

    if not scores:
        return 0.0

    scores.sort(reverse=True)
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]

    weighted_sum = sum(
        sc * (weights[i] if i < len(weights) else 0.01)
        for i, sc in enumerate(scores)
    )
    total_w = sum(weights[:len(scores)]) + 0.01 * max(0, len(scores) - len(weights))

    return min(weighted_sum / total_w, 100.0)

def extract_section_identifier(text: str) -> Optional[str]:
    """Extracts numeric/alphanumeric identifier like 35, 304B, 377A"""
    if not text:
        return None
    m = re.search(r'\b(\d{1,3}[A-Za-z]?)\b', text)
    return m.group(1) if m else None


def clean_act_name(text: str) -> str:
    """Clean act name for fuzzy matching."""
    return remove_connective_fillers(text.lower())


def match_candidate_to_meta(candidate_full: str,
                            id_section_meta: Dict[str, List[str]],
                            threshold: int = TITLE_MATCH_THRESHOLD) -> Optional[Tuple[str, str]]:
    """
    Returns (id, title) matched to candidate_full using hybrid algorithm.
    If no match found, return None.
    """
    section_id = extract_section_identifier(candidate_full)
    if not section_id:
        return None
    
    m = re.search(r'of\s+(.+)', candidate_full, flags=re.IGNORECASE)
    act_part = clean_act_name(m.group(1)) if m else ""

    best_id = None
    best_title = None
    best_score = 0.0

    for meta_key, arr in id_section_meta.items():
        if len(arr) < 4:
            continue

        ref_section = arr[0]
        ref_act     = arr[1]
        ref_title   = arr[2]
        ref_id      = str(arr[3]).strip()

        
        ref_sec_id = extract_section_identifier(ref_section)
        if not ref_sec_id:
            continue

        
        sec_match = (
            ref_sec_id.lower() == section_id.lower()
            or section_id.lower().startswith(ref_sec_id.lower())
        )
        if not sec_match:
            continue

        score = smart_fuzzy_match(act_part, clean_act_name(ref_act))

        if score > best_score and score >= threshold:
            best_score = score
            best_id = ref_id
            best_title = ref_title

    if best_id:
        return best_id, best_title
    return None

def build_id_to_full_map(id_section_meta: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
    """Build id→full mappings."""
    id2full = {}
    for k, arr in id_section_meta.items():
        if not isinstance(arr, list) or len(arr) < 4:
            continue
        section = arr[0]
        act     = arr[1]
        blurb   = arr[2]
        cid     = str(arr[3]).strip()

        full = f"{section} of {act}"

        if cid:
            id2full[cid] = {
                "full": full,
                "section": section,
                "act": act,
                "blurb": blurb,
                "key": k,
            }
    return id2full


def ids_to_full_citations(cand_ids: List[str], id2full: Dict[str, Dict[str, str]]):
    fulls = []
    misses = []
    seen = set()

    for cid in cand_ids:
        meta = id2full.get(str(cid).strip())
        if meta:
            full = meta["full"].strip()
            if full and full not in seen:
                seen.add(full)
                fulls.append(full)
        else:
            misses.append({"id": cid, "reason": "id not found in id-section-meta"})

    return fulls, misses

SYSTEM_RULES = """
You are a legal expert on Indian law. Select statutes that best apply to the user's situation, but ONLY from the provided CANDIDATE_STATUTES.
Rules:
1) Output exactly one list in square brackets using double quotes, e.g., ["Section X of Act";"Section Y of Act"].
2) Each entry must be a quoted string in the exact same form as provided in CANDIDATE_STATUTES (no rephrasing).
3) Do not add anything outside that one list. No commentary.
4) Do not invent or add statutes not in CANDIDATE_STATUTES.
5) Prefer fewer, more precise provisions over many generic ones.
6) If none clearly apply, return [].
"""

def build_prompt_for_model(model_name: str,
                           query_text: str,
                           candidates_full: List[str],
                           defs_map_unused,
                           defs_idx_unused,
                           miss_list_for_query,
                           tokenizer):

    global id_section_meta  
    query_block = f"QUERY:\n{whitespace_handler(query_text)}\n"
    cand_block  = "CANDIDATE_STATUTES:\n" + "\n".join(f'- "{c}"' for c in candidates_full)

    title_lines = []
    for c in candidates_full:
        match = match_candidate_to_meta(c, id_section_meta, TITLE_MATCH_THRESHOLD)
        if match:
            _, title = match
            title_lines.append(f'- "{c}": {whitespace_handler(title)}')
        else:
            title_lines.append(f'- "{c}": (title not found)')
            miss_list_for_query.append({"candidate_full": c, "reason": "title_not_found"})

    titles_block = "STATUTE_TITLES:\n" + "\n".join(title_lines)

    user_msg = (
        f"{SYSTEM_RULES}\n\n"
        f"{query_block}\n"
        f"{cand_block}\n\n"
        f"{titles_block}\n\n"
        f"Return only the bracketed list."
    )

    messages = [{"role": "user", "content": user_msg}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def precompute_train_sbert(
    sbert: SentenceTransformer,
    train_rows: List[Dict[str, Any]],
) -> Tuple[List[str], np.ndarray, List[List[str]]]:
    """
    Returns:
      train_q_texts: List[str]
      train_embeddings: np.ndarray [N, D]
      train_citation_ids_list: List[List[str]]
    """
    print("[INFO] Preparing train queries & citation IDs...")
    tq_texts, tq_ids = [], []
    for row in train_rows:
        tq = whitespace_handler(row.get(TRAIN_QUERY_KEY, ""))
        ids = []
        for key in TRAIN_ID_KEYS:
            raw = row.get(key, [])
            if isinstance(raw, list):
                ids.extend(str(x).strip() for x in raw if str(x).strip())
            else:
                ids.append(str(raw).strip())
        tq_texts.append(tq)
        tq_ids.append(ids)

    print(f"[INFO] Encoding {len(tq_texts)} train queries with SBERT...")
    emb = sbert.encode(
        tq_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print("[INFO] Train embeddings ready.")
    return tq_texts, emb, tq_ids


def build_candidates_ids_from_sbert(
    sbert: SentenceTransformer,
    train_embeddings: np.ndarray,
    train_ids_list: List[List[str]],
    test_query_text: str,
    max_candidates: int = MAX_CANDIDATES,
    topk_sim_train: int = TOPK_SIM_TRAIN,
) -> List[str]:
    """
    For a given test query, retrieve similar train queries via SBERT,
    then accumulate a unique list of citation IDs up to max_candidates.
    """
    q = whitespace_handler(test_query_text)
    q_emb = sbert.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    sims = np.dot(train_embeddings, q_emb[0])  
    idx_sorted = np.argsort(-sims)

    seen = set()
    cand_ids = []

    limit = min(topk_sim_train, len(idx_sorted))
    for i in range(limit):
        ridx = idx_sorted[i]
        for cid in train_ids_list[ridx]:
            cid = str(cid).strip()
            if cid and cid not in seen:
                seen.add(cid)
                cand_ids.append(cid)
                if len(cand_ids) >= max_candidates:
                    return cand_ids
    return cand_ids

def generate_selection(prompt: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> str:
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("[DEBUG] Decoding output...")
    gen = outputs[0][input_len:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def _is_oom_error(e: Exception) -> bool:
    m = str(e).lower()
    return (
        isinstance(e, torch.cuda.OutOfMemoryError)
        or "out of memory" in m
        or "cuda error: out of memory" in m
        or ("cublas" in m and "alloc" in m)
    )

def main(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args):
    global id_section_meta

    print(f"[INFO] Loading TEST from: {TEST_DATA_PATH}")
    test_rows = load_json(TEST_DATA_PATH)
    print(f"[INFO] Loaded {len(test_rows)} test queries.")

    results = []
    miss_log = []
    skips = []

    if os.path.exists(OUTPUT_JSON_PATH):
        print(f"[INFO] Found existing results at {OUTPUT_JSON_PATH}")
        results = load_json(OUTPUT_JSON_PATH)
    if os.path.exists(MISS_LOG_PATH):
        miss_log = load_json(MISS_LOG_PATH)
    if os.path.exists(SKIP_LOG_PATH):
        skips = load_json(SKIP_LOG_PATH)

    done_ids = {r.get("id") for r in results}
    skipped_ids = {s.get("id") for s in skips}

    print(f"[INFO] Loading TRAIN from: {TRAIN_DATA_PATH}")
    train_rows = load_json(TRAIN_DATA_PATH)
    print(f"[INFO] Loaded {len(train_rows)} train rows.")

    print(f"[INFO] Loading ID→SECTION meta from: {ID_SECTION_META_PATH}")
    id_section_meta = load_json(ID_SECTION_META_PATH)
    id2full = build_id_to_full_map(id_section_meta)
    print(f"[INFO] Built ID→full map for {len(id2full)} ids.")

    sbert = load_sbert()
    _, train_embeds, train_ids_list = precompute_train_sbert(sbert, train_rows)

    if args.force_all:
        print("[RESUME] --force-all: reprocessing all.")
        to_process = test_rows
    elif args.rerun_skipped:
        print("[RESUME] --rerun-skipped: only skipped.")
        to_process = [r for r in test_rows if r.get("id") in skipped_ids]
    else:
        print("[RESUME] Default resume.")
        to_process = [r for r in test_rows if r.get("id") not in done_ids]

    print(f"[RESUME] Will process {len(to_process)} query(ies).")
    processed_since_save = 0

    for row in tqdm(to_process, desc="SBERT → ID → Titles → LLM selection"):
        qid = row.get("id")
        print(f"\n[INFO] Processing id={qid} ...")

        query_text = whitespace_handler(row.get(TEST_QUERY_KEY, ""))

        try:
            print("[DEBUG] Building candidate IDs via SBERT...")
            candidate_ids = build_candidates_ids_from_sbert(
                sbert,
                train_embeds,
                train_ids_list,
                query_text,
                MAX_CANDIDATES,
                TOPK_SIM_TRAIN
            )
            print(f"[DEBUG] {len(candidate_ids)} candidate ID(s): {candidate_ids}")
        except Exception as e:
            reason = "SBERTError"
            print(f"[ERROR] {reason}: {e}")
            skips.append({"id": qid, "reason": reason, "error": str(e)})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()

            processed_since_save += 1
            if processed_since_save >= BATCH_SAVE_N:
                save_json(OUTPUT_JSON_PATH, results)
                save_json(MISS_LOG_PATH, miss_log)
                save_json(SKIP_LOG_PATH, skips)
                processed_since_save = 0
            continue

        candidates_full, id_misses = ids_to_full_citations(candidate_ids, id2full)
        print(f"[DEBUG] Resolved {len(candidates_full)} full citations.")

        misses_this = []
        if id_misses:
            misses_this.append({"id_mapping_misses": id_misses})

        print("[DEBUG] Building prompt...")
        prompt = build_prompt_for_model(
            MODEL_ID, query_text, candidates_full,
            None, None,
            misses_this,
            tokenizer
        )

        try:
            raw = generate_selection(prompt, tokenizer, model)
            picked = parse_bracketed_list(raw)
            picked_filtered = [p for p in picked if p in candidates_full]

            print(f"[DEBUG] Selected {len(picked_filtered)} statute(s).")

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
            print(f"[ERROR] {reason}: {e}")
            skips.append({"id": qid, "reason": reason, "error": str(e)})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()

        processed_since_save += 1
        if processed_since_save >= BATCH_SAVE_N:
            print("[INFO] Auto-saving batch...")
            save_json(OUTPUT_JSON_PATH, results)
            save_json(MISS_LOG_PATH, miss_log)
            save_json(SKIP_LOG_PATH, skips)
            processed_since_save = 0

    print("[INFO] Final save...")
    save_json(OUTPUT_JSON_PATH, results)
    save_json(MISS_LOG_PATH, miss_log)
    save_json(SKIP_LOG_PATH, skips)

    print(f"[OK] Saved {len(results)} results.")
    print(f"[OK] Saved {len(miss_log)} miss log entries.")
    print(f"[OK] Saved {len(skips)} skipped queries.")

def build_argparser():
    ap = argparse.ArgumentParser(description="RAG_k_stat_from_QQ-sBert with resume support")

    ap.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing OUTPUT/MISS/SKIP files (default)."
    )

    ap.add_argument(
        "--rerun-skipped",
        action="store_true",
        help="Only re-run the items previously logged in the skip JSON."
    )

    ap.add_argument(
        "--force-all",
        action="store_true",
        help="Ignore saved results and process ALL test queries from scratch."
    )

    return ap

if __name__ == "__main__":
    print("[INFO] Starting pipeline...")
    args = build_argparser().parse_args()

    model, tokenizer = load_model_and_tokenizer()

    main(model, tokenizer, args)

    print("[INFO] Done.")
