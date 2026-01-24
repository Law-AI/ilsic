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

import os, json, re, gc, traceback, numpy as np, argparse
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
import torch

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False

USE_HYBRID_MATCHER = True   
HYBRID_THRESHOLD   = 40     

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
SBERT_DEVICE_NO = 0  

def _resolve_sbert_device() -> str:
    """Decide the exact torch device string for SBERT."""
    env_dev = os.getenv("SBERT_DEVICE", "").strip()  
    if env_dev:
        return env_dev

    if torch.cuda.is_available():
        try:
            return f"cuda:{int(SBERT_DEVICE_NO)}"
        except Exception:
            return "cuda:0"
    return "cpu"

SBERT_DEVICE_STR = _resolve_sbert_device()



TEST_DATA_PATH       = "/home/shounak/HDD/Layman-LSI/dataset/Layman-new-dataset/id-Layman-test-clean.json"
TRAIN_DATA_PATH      = "/home/shounak/HDD/Layman-LSI/dataset/Layman-new-dataset/id-Layman-train-clean.json"
STATUTE_DEFS_PATH    = "/home/shounak/HDD/Layman-LSI/dataset/section-formal-texts.json"
ID_SECTION_META_PATH = "/home/shounak/HDD/Layman-LSI/dataset/id-section-meta.json"


OUTPUT_JSON_PATH     = "/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/RAG/restricted/QQ-sbert/gpt4-1_base_RAG_kStatfrom_QQ.json"
MISS_LOG_PATH        = "/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/RAG/restricted/QQ-sbert/RAG_base_gpt4-1_def_misses_QQ.json"
SKIP_LOG_PATH        = "/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/RAG/restricted/QQ-sbert/RAG_base_gpt4-1_oom_skips_k15_QQ.json"

CACHE_DIR_SBERT      = "/home/shared/hf_cache"


OPENAI_API_KEY   = "OpenAI API key"  
OPENAI_MODEL     = "gpt-4.1"            
TEMPERATURE      = 0.0
MAX_OUTPUT_TOKENS = 1024              
BATCH_SAVE_N     = 10


SBERT_MODEL_ID       = "sentence-transformers/all-mpnet-base-v2"
MAX_CANDIDATES       = 15     
TOPK_SIM_TRAIN       = 500    
SBERT_DEVICE         = SBERT_DEVICE_STR  

TEST_QUERY_KEY       = "query-text"
TRAIN_QUERY_KEY      = "query-text"
TRAIN_ID_KEYS        = ["citations-id"]  

def _print_cuda_debug_banner():
    print(f"[CUDA] torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"[CUDA] CUDA_VISIBLE_DEVICES = {os.getenv('CUDA_VISIBLE_DEVICES')}")
    if torch.cuda.is_available():
        print(f"[CUDA] torch.cuda.device_count() = {torch.cuda.device_count()}")
        try:
            cur = torch.cuda.current_device()
            print(f"[CUDA] torch.cuda.current_device() = {cur}")
            print(f"[CUDA] torch.cuda.get_device_name({cur}) = {torch.cuda.get_device_name(cur)}")
        except Exception as e:
            print(f"[CUDA] Unable to query current device name: {e}")
    print(f"[CUDA] Chosen SBERT_DEVICE_STR = {SBERT_DEVICE_STR}")

def load_sbert() -> "SentenceTransformer":
    if not SBERT_AVAILABLE:
        raise RuntimeError("Sentence-BERT not installed. `pip install sentence-transformers`")

    _print_cuda_debug_banner()
    print(f"[INFO] Loading SBERT model: {SBERT_MODEL_ID}")

    sbert = SentenceTransformer(SBERT_MODEL_ID, device="cpu", cache_folder=CACHE_DIR_SBERT)

    if SBERT_DEVICE_STR.lower().startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(SBERT_DEVICE_STR.split(":")[1]))  # set current device
        except Exception:
            pass
        try:
            sbert = sbert.to(SBERT_DEVICE_STR)
        except Exception as e:
            print(f"[WARN] Could not move SBERT to {SBERT_DEVICE_STR}: {e}. Falling back to CPU.")
            sbert = sbert.to("cpu")
    else:
        sbert = sbert.to("cpu")

    try:
        print(f"[INFO] SBERT target device = {getattr(sbert, 'device', 'unknown')}")
    except Exception:
        pass

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

def _norm_text_light(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    s = s.replace(' of the ', ' of ')
    s = re.sub(r'[^a-z0-9 ]+', '', s)
    return s.strip()

def _norm_text_heavy(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\bthe(?=indian\b)', 'the ', s)  
    s = re.sub(r'\s+', ' ', s)
    s = s.replace(' of the ', ' of ')
    s = re.sub(r'\bthe\b', ' ', s)              
    s = re.sub(r'\b\d{4}\b', ' ', s)           
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)          
    s = re.sub(r'\s+', ' ', s).strip()
    return s.replace(' ', '')                   

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
    """Collect citation IDs (strings) from row across possible keys."""
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
1) Output exactly one list in square brackets using double quotes, e.g., ["Section X of Act";"Section Y of Act"].
2) Each entry must be a quoted string in the exact same form as provided in CANDIDATE_STATUTES (no rephrasing).
3) Do not add anything outside that one list. No commentary.
4) Do not invent or add statutes not in CANDIDATE_STATUTES.
5) Prefer fewer, more precise provisions over many generic ones.
6) If none clearly apply, return [].
"""

def build_messages_for_openai(
    query_text: str,
    candidates_full: List[str],
    defs_map: Optional[Dict[str, str]],
    defs_idx: Optional[Dict[str, str]],
    miss_list_for_query: List[Any]
) -> List[Dict[str, str]]:
    query_block = f"QUERY:\n{whitespace_handler(query_text)}\n"
    cand_block  = "CANDIDATE_STATUTES:\n" + "\n".join([f'- "{c}"' for c in candidates_full])

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

    user_msg = f"{query_block}\n{cand_block}\n\n{defs_block}\n\nReturn only the bracketed list."
    return [
        {"role": "system", "content": SYSTEM_RULES.strip()},
        {"role": "user", "content": user_msg.strip()},
    ]

def precompute_train_sbert(
    sbert: SentenceTransformer,
    train_rows: List[Dict[str, Any]],
) -> Tuple[List[str], np.ndarray, List[List[str]]]:
    """
    Returns:
      train_q_texts: List[str]
      train_embeddings: np.ndarray [N, D]
      train_citation_ids_list: List[List[str]]  # list of ID strings per row
    """
    print("[INFO] Preparing train queries & citation IDs...")
    tq_texts, tq_ids = [], []
    for row in train_rows:
        tq = extract_query_text(row, TRAIN_QUERY_KEY)
        ids = extract_id_list(row, TRAIN_ID_KEYS)
        tq_texts.append(tq)
        tq_ids.append(ids)

    print(f"[INFO] Encoding {len(tq_texts)} train queries with SBERT...")
    emb = sbert.encode(
        tq_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
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

    seen, cand_ids = set(), []
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

def build_id_to_full_map(id_section_meta: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
    """
    Build an inverted map from numeric/string ID (value[index=3]) to:
      {
        "full": f"{index0} of {index1}",
        "section": index0,
        "act": index1,
        "blurb": index2,
        "key": original_key
      }
    """
    id2full: Dict[str, Dict[str, str]] = {}
    for k, arr in id_section_meta.items():
        if not isinstance(arr, list) or len(arr) < 4:
            continue
        section = arr[0]
        act     = arr[1]
        blurb   = arr[2]
        cid     = str(arr[3]).strip()  
        full    = f"{section} of {act}"
        if cid:
            id2full[cid] = {
                "full": full,
                "section": section,
                "act": act,
                "blurb": blurb,
                "key": k,
            }
    return id2full

def ids_to_full_citations(cand_ids: List[str], id2full: Dict[str, Dict[str, str]]) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Convert candidate IDs to full-citation strings.
    Returns (full_list, misses) where misses hold ids not found in meta.
    """
    fulls, misses = [], []
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

def parse_raw_generation_flexible(raw_text: str) -> List[str]:
    """
    Parse raw_generation that may be:
      - a JSON array string with commas
      - a semicolon-separated pseudo-list
      - a quoted, comma-separated pseudo-list (not valid JSON)
    Returns list[str].
    """
    if not isinstance(raw_text, str):
        return []
    s = raw_text.strip()

    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    if s.startswith("[") and s.endswith("]"):
        s_inner = s[1:-1].strip()
    else:
        s_inner = s

    if ";" in s_inner:
        parts = re.split(r"\s*;\s*", s_inner)
    else:
        parts = re.split(r'\s*,\s*(?=["\'])', s_inner)
    items = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
            p = p[1:-1]
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            items.append(p)
    return items

def map_raw_to_candidates(raw_generation: str,
                          candidates_full: List[str],
                          fuzzy_threshold: int = 80) -> Tuple[List[str], List[str]]:
    """
    Robustly map model's free-form raw_generation to canonical entries in candidates_full.
    1) Parse raw_generation tolerantly (JSON or semi-colon / malformed quoted lists)
    2) Try exact normalized match (light)
    3) Try exact normalized match (heavy: drop years, 'the', punctuation, spaces)
    4) Fallback to fuzzy match (RapidFuzz token_set_ratio) on heavy-normalized strings
    Returns (selected_list, unmatched_raw_items)
    """
    picked = parse_raw_generation_flexible(raw_generation)

    norm_light_to_cand = {_norm_text_light(c): c for c in candidates_full}
    norm_heavy_to_cand = {_norm_text_heavy(c): c for c in candidates_full}
    cand_heavy_list    = list(norm_heavy_to_cand.keys())

    selected, unmatched = [], []

    for p in picked:
        pl = _norm_text_light(p)
        if pl in norm_light_to_cand:
            cand = norm_light_to_cand[pl]
            if cand not in selected:
                selected.append(cand)
            continue

        ph = _norm_text_heavy(p)
        if ph in norm_heavy_to_cand:
            cand = norm_heavy_to_cand[ph]
            if cand not in selected:
                selected.append(cand)
            continue

        if RAPIDFUZZ_AVAILABLE and cand_heavy_list:
            best, best_score = None, 0
            for ch in cand_heavy_list:
                sc = fuzz.token_set_ratio(ph, ch)
                if sc > best_score:
                    best, best_score = ch, sc
            if best is not None and best_score >= fuzzy_threshold:
                cand = norm_heavy_to_cand[best]
                if cand not in selected:
                    selected.append(cand)
            else:
                unmatched.append(p)
        else:
            unmatched.append(p)

    return selected, unmatched

CONNECTIVE_FILLERS = {
    "under","of","by","on","in","at","with","and","for","from","to","a","an","or","but","as","per",
    "via","upon","within","into","through","over","between","among","around","about","before","after",
    "during","since","until","towards","against","without","including","regarding","concerning",
    "according","versus","vs","w.r.t",
}

def _hm_remove_connective_fillers(text: str) -> str:
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

def _hm_normalize_identifier_token(token: str) -> str:
    result = []
    for c in token:
        if c in "([":  
            break
        if c.isalnum():
            result.append(c)
    return "".join(result)

def _hm_analyze_text_structure(text: str) -> Dict:
    if not text:
        return {}
    words = text.split()
    analysis = {
        "words": words, "word_count": len(words),
        "numeric_words": [], "short_alphanum_words": [], "long_words": [],
        "potential_identifiers": [],
    }
    for i, word in enumerate(words):
        normalized = _hm_normalize_identifier_token(word)
        if normalized.isdigit():
            analysis["numeric_words"].append((i, normalized))
        elif len(normalized) <= 5 and any(c.isdigit() for c in normalized):
            analysis["short_alphanum_words"].append((i, normalized))
        elif len(normalized) > 8:
            analysis["long_words"].append((i, normalized))
        if len(word) <= 6 and (any(c.isdigit() for c in normalized) or any(x in word for x in "()[].")):
            analysis["potential_identifiers"].append((i, word))
    return analysis

def _hm_find_section_identifier(text: str) -> Tuple[Optional[str], int]:
    analysis = _hm_analyze_text_structure(text)
    words = analysis.get("words", [])
    for pos, num in analysis.get("numeric_words", []):
        if pos <= 2:
            if pos + 1 < len(words):
                nxt = words[pos + 1]
                if len(nxt) == 1 and nxt.isalpha():
                    return num + nxt, pos
            return num, pos
    for pos, word in analysis.get("short_alphanum_words", []):
        if pos <= 3:
            return word, pos
    for pos, word in analysis.get("potential_identifiers", []):
        if pos <= 3:
            cleaned = _hm_normalize_identifier_token(word)
            if cleaned:
                return cleaned, pos
    return None, -1

def _hm_intelligent_split_section_act(text: str) -> Tuple[Optional[str], Optional[str]]:
    text = text.strip()
    if not text:
        return None, None
    words = text.split()
    if len(words) < 2:
        return text, ""
    identifier, pos = _hm_find_section_identifier(text)
    if identifier and pos >= 0:
        section_words = []
        if pos > 0:
            prev_word = words[pos - 1]
            if len(prev_word) <= 10 and not any(c.isdigit() for c in prev_word):
                section_words.append(prev_word)
        section_words.append(identifier)
        section = " ".join(section_words).strip()
        used = set(w.lower() for w in section_words)
        remaining = [w for w in words if w.lower() not in used]
        act = " ".join(remaining).strip()
        return section, act
    return None, text

def _hm_extract_pairs(output_text) -> List[Tuple[str, str]]:
    if isinstance(output_text, str):
        s = output_text.strip().strip("[]")
        s = re.sub(r'^"+|"+$', '', s)
        top = [item.strip().strip("\"'") for item in re.split(r"\s*;\s*", s) if item.strip()]
        raw_items = []
        for item in top:
            raw_items.extend(re.split(r"""\s*,\s*(?=(?:[Ss](?:ection|ec)\.?|\d{1,3}[A-Za-z]?)\b)""", item))
        items = [i.strip().strip("\"'") for i in raw_items if i.strip()]
    elif isinstance(output_text, list):
        items = []
        for it in output_text:
            items.extend(re.split(r"""\s*,\s*(?=(?:[Ss](?:ection|ec)\.?|\d{1,3}[A-Za-z]?)\b)""", it))
        items = [i.strip().strip("\"'") for i in items if i.strip()]
    else:
        return []

    pairs = []
    for pair in items:
        sec, act = _hm_intelligent_split_section_act(pair)
        if sec:
            pairs.append((sec, act if act else ""))
    return pairs

def _hm_normalize_for_comparison(text: str) -> str:
    if not text:
        return ""
    text = _hm_remove_connective_fillers(text)
    words = text.split()
    filtered = []
    for word in words:
        clean = "".join(c for c in word if c.isalnum())
        if any(c.isdigit() for c in clean) or len(clean) > 2 or (len(clean) == 2 and clean.isalpha()):
            filtered.append(clean)
    if len(filtered) < len(words) / 3:
        return " ".join("".join(c for c in w if c.isalnum()) for w in words if w)
    return " ".join(filtered)

def _hm_smart_fuzzy_match(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0

    clean1 = _hm_remove_connective_fillers(text1)
    clean2 = _hm_remove_connective_fillers(text2)
    norm1 = _hm_normalize_for_comparison(clean1)
    norm2 = _hm_normalize_for_comparison(clean2)

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
            scores.append((inter / uni) * 100.0)

    word_scores = []
    for a in w1:
        for b in w2:
            if a == b:
                word_scores.append(len(a) * 10.0)
            elif len(a) > 3 and len(b) > 3:
                sim = fuzz.ratio(a, b)
                if sim > 80:
                    word_scores.append(sim * len(a) / 10.0)
    if word_scores:
        avg_word = sum(word_scores) / len(word_scores)
        scores.append(min(avg_word, 100.0))

    if not scores:
        return 0.0

    scores.sort(reverse=True)

    weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    weighted_sum = 0.0
    for i, sc in enumerate(scores):
        w = weights[i] if i < len(weights) else 0.01 
        weighted_sum += sc * w
    total_w = sum(weights[:min(len(weights), len(scores))]) + 0.01 * max(0, len(scores) - len(weights))

    return min(weighted_sum / total_w, 100.0)


def _hm_get_matched_ids(pairs: List[Tuple[str, str]], reference_data: Dict[str, List[str]], threshold: int) -> List[Tuple[str, str]]:
    """Return list of (matched_id or 'NA', original_pair_string) preserving order."""
    out: List[Tuple[str, str]] = []
    for section_input, act_input in pairs:
        input_id, _ = _hm_find_section_identifier(section_input)
        best_score = 0.0
        best_id: Optional[str] = None

        for _, ref_value in reference_data.items():
            if len(ref_value) < 4:
                continue
            ref_section, ref_act, _, ref_id = ref_value[0], ref_value[1], ref_value[2], ref_value[3]
            ref_identifier, _ = _hm_find_section_identifier(ref_section)
            if not ref_identifier or not input_id:
                continue
            low_in, low_ref = input_id.lower(), ref_identifier.lower()
            matched_key = (low_in == low_ref) or (len(low_in) > 1 and low_in[-1].isalpha() and low_in[:-1] == low_ref)
            if not matched_key:
                continue
            clean_input_act = _hm_remove_connective_fillers(act_input.lower()) if act_input else ""
            clean_ref_act = _hm_remove_connective_fillers(ref_act.lower()) if ref_act else ""
            act_score = (
                _hm_smart_fuzzy_match(clean_input_act, clean_ref_act) if clean_input_act and clean_ref_act
                else (100.0 if not clean_input_act and not clean_ref_act else 70.0)
            )
            if act_score > best_score:
                best_score = act_score
                best_id = str(ref_id)
        out.append((best_id if best_id and best_score >= threshold else "NA", f"{section_input} {act_input}".strip()))
    return out

def _hybrid_map_raw_to_selected(
    raw_generation: str,
    reference_data: Dict[str, List[str]],
    id2full: Dict[str, Dict[str, str]],
    candidates_full: List[str],
    threshold: int
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns:
      inside_candidates_full: mapped full strings that are in candidates_full
      outside_candidates_full: mapped full strings not in candidates_full
      unmatched_raw: raw items we couldn't map to any id
    """
    pairs = _hm_extract_pairs(raw_generation)
    matched = _hm_get_matched_ids(pairs, reference_data, threshold)

    inside, outside, unmatched = [], [], []
    cand_set = set(candidates_full)
    for mid, raw_item in matched:
        if mid == "NA":
            unmatched.append(raw_item)
            continue
        meta = id2full.get(str(mid))
        if not meta:
            unmatched.append(raw_item)
            continue
        full = meta["full"].strip()
        if full in cand_set:
            if full not in inside:
                inside.append(full)
        else:
            if full not in outside:
                outside.append(full)
    return inside, outside, unmatched


from openai import OpenAI
_openai_client: Optional[OpenAI] = None

def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY or OPENAI_API_KEY == "PUT_YOUR_KEY_HERE":
            raise RuntimeError("Please set OPENAI_API_KEY at the top of the script.")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

def generate_selection_via_openai(messages: List[Dict[str, str]]) -> str:
    """
    Calls OpenAI Chat Completions with GPT-4.1 (or 4.1-mini).
    Returns raw text (the model's content).
    """
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()


def main(args):
    print(f"[INFO] Loading TEST from: {TEST_DATA_PATH}")
    test_rows = load_json(TEST_DATA_PATH)
    print(f"[INFO] Loaded {len(test_rows)} test queries.")


    results: List[Dict[str, Any]] = []
    miss_log: List[Dict[str, Any]] = []
    skips: List[Dict[str, Any]] = []

    if os.path.exists(OUTPUT_JSON_PATH):
        print(f"[INFO] Found existing results at {OUTPUT_JSON_PATH}")
        results = load_json(OUTPUT_JSON_PATH) or []
    if os.path.exists(MISS_LOG_PATH):
        miss_log = load_json(MISS_LOG_PATH) or []
    if os.path.exists(SKIP_LOG_PATH):
        skips = load_json(SKIP_LOG_PATH) or []

    done_ids = {r.get("id") for r in results}
    skipped_ids = {s.get("id") for s in skips}


    print(f"[INFO] Loading TRAIN from: {TRAIN_DATA_PATH}")
    train_rows = load_json(TRAIN_DATA_PATH)
    print(f"[INFO] Loaded {len(train_rows)} train rows.")

    print(f"[INFO] Loading ID→SECTION meta from: {ID_SECTION_META_PATH}")
    id_section_meta = load_json(ID_SECTION_META_PATH)
    id2full = build_id_to_full_map(id_section_meta)
    print(f"[INFO] Built ID→full map for {len(id2full)} ids.")
    reference_data = id_section_meta  # (UPDATED) reuse as hybrid reference

    defs = load_json(STATUTE_DEFS_PATH) if STATUTE_DEFS_PATH else None
    defs_idx = build_defs_index(defs) if defs else None
    if defs:
        print(f"[INFO] Loaded {len(defs)} statute definitions. (normalized index ready)")

    sbert = load_sbert()
    _, train_embeds, train_ids_list = precompute_train_sbert(sbert, train_rows)

    if args.force_all:
        print("[RESUME] --force-all specified: ignoring previous outputs; reprocessing all test queries.")
        to_process = test_rows
    elif args.rerun_skipped:
        print("[RESUME] --rerun-skipped specified: only re-running previously skipped queries.")
        to_process = [row for row in test_rows if isinstance(row, dict) and row.get("id") in skipped_ids]
    else:
        print("[RESUME] Default resume: skipping already completed ids and continuing remaining.")
        to_process = [row for row in test_rows if isinstance(row, dict) and row.get("id") not in done_ids]

    print(f"[RESUME] Will process {len(to_process)} query(ies).")
    processed_since_save = 0

    for row in tqdm(to_process, desc="SBERT→ID candidates→full→GPT-4.1 selection"):
        qid = row.get("id")
        print(f"\n[INFO] Processing test id={qid} ...")

        query_text = extract_query_text(row, TEST_QUERY_KEY)

        try:
            print("[DEBUG] Building candidate IDs via SBERT retrieval...")
            candidate_ids = build_candidates_ids_from_sbert(
                sbert=sbert,
                train_embeddings=train_embeds,
                train_ids_list=train_ids_list,
                test_query_text=query_text,
                max_candidates=MAX_CANDIDATES,
                topk_sim_train=TOPK_SIM_TRAIN,
            )
            print(f"[DEBUG] {len(candidate_ids)} candidate ID(s): {candidate_ids}")
        except Exception as e:
            reason = "SBERTError"
            tb_tail = "\n".join(traceback.format_exc().splitlines()[-5:])
            print(f"[ERROR] {reason} on id={qid}. Skipping. Details:\n{tb_tail}")
            try:
                pass  
            except Exception:
                pass
            gc.collect()
            skips.append({"id": qid, "reason": reason, "error": str(e)})
            processed_since_save += 1
            if processed_since_save >= BATCH_SAVE_N:
                print(f"[INFO] Auto-saving (after errors) ...")
                save_json(OUTPUT_JSON_PATH, results)
                save_json(MISS_LOG_PATH, miss_log)
                save_json(SKIP_LOG_PATH, skips)
                processed_since_save = 0
            continue

        candidates_full, id_misses = ids_to_full_citations(candidate_ids, id2full)
        print(f"[DEBUG] Resolved {len(candidates_full)} full citation(s).")
        misses_this: List[Any] = []
        if id_misses:
            misses_this.append({"id_mapping_misses": id_misses})

        print("[DEBUG] Building messages for OpenAI...")
        messages = build_messages_for_openai(
            query_text=query_text,
            candidates_full=candidates_full,
            defs_map=defs,
            defs_idx=defs_idx,
            miss_list_for_query=misses_this
        )

        try:
            raw = generate_selection_via_openai(messages)

            if USE_HYBRID_MATCHER:
                picked_filtered, outside_full, unmatched = _hybrid_map_raw_to_selected(
                    raw_generation=raw,
                    reference_data=reference_data,
                    id2full=id2full,
                    candidates_full=candidates_full,
                    threshold=HYBRID_THRESHOLD,
                )
                selected_outside_candidates = outside_full[:]  
            else:
                picked_filtered, unmatched = map_raw_to_candidates(
                    raw_generation=raw,
                    candidates_full=candidates_full,
                    fuzzy_threshold=80,
                )
                selected_outside_candidates = unmatched[:]  

            print(f"[DEBUG] Selected {len(picked_filtered)} statute(s).")

            results.append({
                "id": qid,
                "input": query_text,
                "candidate_ids": candidate_ids,
                "candidate_statutes": candidates_full,
                "selected_statutes": picked_filtered,
                "selected_outside_candidates": selected_outside_candidates,  
                "raw_generation": raw
            })

            if unmatched:
                misses_this.append({"raw_unmatched": unmatched})

            if misses_this:
                miss_log.append({
                    "id": qid,
                    "missing": misses_this
                })

        except Exception as e:
            reason = "OtherError"
            tb_tail = "\n".join(traceback.format_exc().splitlines()[-5:])
            print(f"[ERROR] {reason} on id={qid}. Skipping. Details:\n{tb_tail}")
            gc.collect()
            skips.append({
                "id": qid,
                "reason": reason,
                "error": str(e),
            })

        processed_since_save += 1
        if processed_since_save >= BATCH_SAVE_N:
            print(f"[INFO] Auto-saving after batch...")
            save_json(OUTPUT_JSON_PATH, results)
            save_json(MISS_LOG_PATH, miss_log)
            save_json(SKIP_LOG_PATH, skips)
            processed_since_save = 0

    print("[INFO] Final save...")
    save_json(OUTPUT_JSON_PATH, results)
    save_json(MISS_LOG_PATH, miss_log)
    save_json(SKIP_LOG_PATH, skips)
    print(f"[OK] Saved {len(results)} results to {OUTPUT_JSON_PATH}")
    print(f"[OK] Saved {len(miss_log)} miss-log entries to {MISS_LOG_PATH}")
    print(f"[OK] Saved {len(skips)} skipped queries to {SKIP_LOG_PATH}")


def build_argparser():
    ap = argparse.ArgumentParser(description="RAG_k_stat_from_QQ-sBert with resume support (OpenAI GPT-4.1)")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Resume from existing OUTPUT/MISS/SKIP files (default).")
    ap.add_argument("--rerun-skipped", action="store_true",
                    help="Only re-run the items previously logged in skips JSON.")
    ap.add_argument("--force-all", action="store_true",
                    help="Ignore saved results and process all test queries from scratch.")
    return ap

if __name__ == "__main__":
    print("[INFO] Starting pipeline (OpenAI GPT-4.1)...")
    args = build_argparser().parse_args()
    main(args)
    print("[INFO] Done.")
