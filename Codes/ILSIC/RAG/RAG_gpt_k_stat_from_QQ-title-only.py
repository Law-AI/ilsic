"""
===========================================================
 RAG_k_stat_from_QQ-sBert — GPT-4.1 (TITLE-ONLY RAG)
===========================================================

This is a faithful port of the LLaMA title-only RAG pipeline:
- SBERT query→query retrieval
- ID → full citation mapping
- Hybrid fuzzy title matching via id-section-meta
- STATUTE_TITLES injected into the prompt
- GPT used ONLY for constrained selection

===========================================================
"""

import os, json, re, gc, argparse, traceback
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import torch

from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from openai import OpenAI

DEFAULT_OPENAI_API_KEY = "OPENAI API KEY"   
DEFAULT_SBERT_DEVICE  = "cuda:1"

TEST_DATA_PATH = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/Layman-new-dataset/id-Layman-test-clean-new.json"
TRAIN_DATA_PATH = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/Layman-new-dataset/id-Layman-train-clean-new.json"
ID_SECTION_META_PATH = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/id-section-meta.json"

OUTPUT_JSON_PATH = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/Rebuttal_ARR_Oct_2025/restricted/title-only/RAG_gpt_4-1_kStatfrom_title-only_QQ.json"
MISS_LOG_PATH    = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/Rebuttal_ARR_Oct_2025/restricted/title-only/RAG_gpt_4-1_title_misses.json"
SKIP_LOG_PATH    = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/Rebuttal_ARR_Oct_2025/restricted/title-only/RAG_gpt_4-1_skips.json"

CACHE_DIR_SBERT = "/home/shared/hf_cache"

OPENAI_MODEL = "gpt-4.1"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 1024

SBERT_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MAX_CANDIDATES = 15
TOPK_SIM_TRAIN = 500
BATCH_SAVE_N = 10

TITLE_MATCH_THRESHOLD = 70

TEST_QUERY_KEY  = "query-text"
TRAIN_QUERY_KEY = "query-text"
TRAIN_ID_KEYS   = ["citations-id"]

def get_openai_api_key():
    return os.getenv("OPENAI_API_KEY") or DEFAULT_OPENAI_API_KEY

def resolve_sbert_device():
    return os.getenv("SBERT_DEVICE") or (
        DEFAULT_SBERT_DEVICE if torch.cuda.is_available() else "cpu"
    )

def whitespace_handler(text: str) -> str:
    text = re.sub(r"\n+", " ", str(text).strip())
    text = re.sub(r"\s+", " ", text)
    return text.replace("\xad", "")

def parse_bracketed_list(text: str) -> List[str]:
    m = re.search(r'\[(.*)\]', text, flags=re.DOTALL)
    if not m:
        return []
    return [x.strip() for x in re.findall(r'"([^"]+)"', m.group(1))]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_sbert():
    device = resolve_sbert_device()
    print(f"[INFO] Loading SBERT on {device}")
    sbert = SentenceTransformer(
        SBERT_MODEL_ID,
        device=device,
        cache_folder=CACHE_DIR_SBERT
    )
    sbert.eval()
    return sbert

def precompute_train_sbert(sbert, train_rows):
    texts, ids = [], []
    for row in train_rows:
        texts.append(whitespace_handler(row.get(TRAIN_QUERY_KEY, "")))
        ids.append([str(x).strip() for x in row.get(TRAIN_ID_KEYS[0], [])])
    emb = sbert.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return emb, ids

def build_candidates_ids_from_sbert(train_embeds, train_ids, query):
    q_emb = sbert.encode([query], normalize_embeddings=True)[0]
    sims = np.dot(train_embeds, q_emb)
    idx = np.argsort(-sims)

    seen, out = set(), []
    for i in idx[:TOPK_SIM_TRAIN]:
        for cid in train_ids[i]:
            if cid and cid not in seen:
                seen.add(cid)
                out.append(cid)
                if len(out) >= MAX_CANDIDATES:
                    return out
    return out

CONNECTIVE_FILLERS = {
    "under","of","by","on","in","at","with","and","for","from","to","a","an","or",
    "but","as","per","via","upon","within","into","through","over","between",
    "among","around","about","before","after","during","since","until","towards",
    "against","without","including","regarding","concerning","according","versus",
    "vs","w.r.t",
}

def remove_connective_fillers(text: str) -> str:
    words = text.lower().split()
    return " ".join(
        "".join(c for c in w if c.isalnum())
        for w in words
        if w not in CONNECTIVE_FILLERS or any(c.isdigit() for c in w)
    )

def normalize_for_comparison(text: str) -> str:
    text = remove_connective_fillers(text)
    return " ".join(
        w for w in text.split()
        if any(c.isdigit() for c in w) or len(w) > 2
    )

def smart_fuzzy_match(text1: str, text2: str) -> float:
    clean1, clean2 = remove_connective_fillers(text1), remove_connective_fillers(text2)
    norm1, norm2 = normalize_for_comparison(clean1), normalize_for_comparison(clean2)
    scores = [
        fuzz.ratio(clean1, clean2),
        fuzz.token_sort_ratio(clean1, clean2),
        fuzz.token_set_ratio(clean1, clean2),
        fuzz.ratio(norm1, norm2),
        fuzz.token_set_ratio(norm1, norm2),
    ]
    return max(scores)

def extract_section_identifier(text: str) -> Optional[str]:
    m = re.search(r'\b(\d{1,3}[A-Za-z]?)\b', text)
    return m.group(1) if m else None

def match_candidate_to_meta(candidate_full: str,
                            id_section_meta: Dict[str, List[str]],
                            threshold: int):
    section_id = extract_section_identifier(candidate_full)
    if not section_id:
        return None

    m = re.search(r'of\s+(.+)', candidate_full, flags=re.I)
    act_part = m.group(1).lower() if m else ""

    best_score, best_title = 0.0, None

    for arr in id_section_meta.values():
        if len(arr) < 4:
            continue
        ref_section, ref_act, ref_title = arr[0], arr[1], arr[2]
        ref_id = extract_section_identifier(ref_section)
        if not ref_id or ref_id.lower() != section_id.lower():
            continue

        score = smart_fuzzy_match(act_part, ref_act.lower())
        if score >= threshold and score > best_score:
            best_score, best_title = score, ref_title

    return best_title

SYSTEM_RULES = """
You are a legal expert on Indian law.
Select statutes that best apply to the user's situation, but ONLY from the provided CANDIDATE_STATUTES.
Rules:
1) Output exactly one list in square brackets using double quotes.
2) Each entry must exactly match a string in CANDIDATE_STATUTES.
3) Do not add commentary.
4) If none apply, return [].
"""

def build_messages(query_text, candidates_full, id_section_meta, miss_list):
    title_lines = []
    for c in candidates_full:
        title = match_candidate_to_meta(c, id_section_meta, TITLE_MATCH_THRESHOLD)
        if title:
            title_lines.append(f'- "{c}": {whitespace_handler(title)}')
        else:
            title_lines.append(f'- "{c}": (title not found)')
            miss_list.append({"candidate_full": c, "reason": "title_not_found"})

    user_msg = (
        f"{SYSTEM_RULES}\n\n"
        f"QUERY:\n{query_text}\n\n"
        f"CANDIDATE_STATUTES:\n" +
        "\n".join(f'- "{c}"' for c in candidates_full) +
        "\n\nSTATUTE_TITLES:\n" +
        "\n".join(title_lines) +
        "\n\nReturn only the bracketed list."
    )

    return [{"role": "user", "content": user_msg}]

_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        key = get_openai_api_key()
        if not key or key.startswith("sk-xxxxxxxx"):
            raise RuntimeError("OpenAI API key not set.")
        _openai_client = OpenAI(api_key=key)
    return _openai_client

def generate_selection(messages):
    client = get_openai_client()
    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS
    )
    return (r.choices[0].message.content or "").strip()

def main(args):
    test_rows = load_json(TEST_DATA_PATH)
    train_rows = load_json(TRAIN_DATA_PATH)
    id_section_meta = load_json(ID_SECTION_META_PATH)

    results = load_json(OUTPUT_JSON_PATH) if os.path.exists(OUTPUT_JSON_PATH) else []
    miss_log = load_json(MISS_LOG_PATH) if os.path.exists(MISS_LOG_PATH) else []
    skips = load_json(SKIP_LOG_PATH) if os.path.exists(SKIP_LOG_PATH) else []

    done_ids = {r["id"] for r in results}

    to_process = (
        test_rows if args.force_all else
        [r for r in test_rows if r["id"] not in done_ids]
    )

    global sbert
    sbert = load_sbert()
    train_embeds, train_ids = precompute_train_sbert(sbert, train_rows)

    since = 0
    for row in tqdm(to_process, desc="SBERT → ID → TITLES → GPT"):
        qid = row["id"]
        try:
            query = whitespace_handler(row[TEST_QUERY_KEY])
            cand_ids = build_candidates_ids_from_sbert(train_embeds, train_ids, query)
            candidates_full = []
            for cid in cand_ids:
                for arr in id_section_meta.values():
                    if str(arr[3]) == cid:
                        candidates_full.append(f"{arr[0]} of {arr[1]}")

            miss_list = []
            messages = build_messages(query, candidates_full, id_section_meta, miss_list)
            raw = generate_selection(messages)
            picked = [p for p in parse_bracketed_list(raw) if p in candidates_full]

            results.append({
                "id": qid,
                "input": query,
                "candidate_ids": cand_ids,
                "candidate_statutes": candidates_full,
                "selected_statutes": picked,
                "raw_generation": raw
            })

            if miss_list:
                miss_log.append({"id": qid, "missing": miss_list})

        except Exception as e:
            skips.append({"id": qid, "error": str(e)})
            gc.collect()

        since += 1
        if since >= BATCH_SAVE_N:
            save_json(OUTPUT_JSON_PATH, results)
            save_json(MISS_LOG_PATH, miss_log)
            save_json(SKIP_LOG_PATH, skips)
            since = 0

    save_json(OUTPUT_JSON_PATH, results)
    save_json(MISS_LOG_PATH, miss_log)
    save_json(SKIP_LOG_PATH, skips)

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-all", action="store_true")
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
