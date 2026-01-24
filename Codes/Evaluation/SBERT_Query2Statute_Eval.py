"""
SBERT Retrieval & K@{1..K_MAX} Evaluation for Statute Prediction
- Gold IDs are read from key: "citations-id"
- Saves per-query rankings to JSONL, flushing every 10 queries
- Evaluates AFTER all queries are processed
- Outputs a single compact table with rows:
    mP, muP, mR, muR, mF1, muF1
  and columns:
    k=1 ... k=K_MAX
  * Printed to terminal
  * Saved as CSV (one file)
  * A summary JSON is saved as well

Deps:
  pip install sentence-transformers numpy torch tqdm pandas
"""

import os
import json
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"


QUERIES_JSON       = "/home/shounak/HDD/Layman-LSI/dataset/Layman-new-dataset/id-Layman-test-clean.json"   
NAME_TO_DEF_JSON   = "/home/shounak/HDD/Layman-LSI/dataset/section-formal-texts.json"                      
ID_TO_NAME_JSON    = "/home/shounak/HDD/Layman-LSI/dataset/id-section-meta.json"                           

OUTPUT_DIR             = "/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/analysis/sbert_at_15"
OUTPUT_RANKINGS_JSONL  = os.path.join(OUTPUT_DIR, "rankings_per_query.jsonl")  # append every 10
OUTPUT_SUMMARY_JSON    = os.path.join(OUTPUT_DIR, "eval_summary.json")
OUTPUT_K_TABLE_CSV     = os.path.join(OUTPUT_DIR, "k_eval_table_mP_muP_mR_muR_mF1_muF1.csv")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE_STATUTES = 1024
BATCH_SIZE_QUERIES  = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K_MAX_DEFAULT = 15

QUERY_ID_KEY    = "id"
QUERY_TEXT_KEY  = "query-text"
GOLD_KEY        = "citations-id"   

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def join_section_act(section: str, act: str) -> str:
    section = (section or "").strip()
    act = (act or "").strip()
    if section and act:
        if not section.lower().startswith("section"):
            section = f"Section {section}"
        return f"{section} of {act}"
    return section or act

def extract_query(q: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    if QUERY_ID_KEY not in q:
        raise ValueError(f"Query object missing '{QUERY_ID_KEY}': {q}")
    if QUERY_TEXT_KEY not in q:
        raise ValueError(f"Query object missing '{QUERY_TEXT_KEY}': {q}")

    qid = str(q[QUERY_ID_KEY])
    qtext = str(q[QUERY_TEXT_KEY]).strip()

    gold_raw = q.get(GOLD_KEY, [])
    if gold_raw is None:
        gold_ids = []
    elif isinstance(gold_raw, (list, tuple)):
        gold_ids = [str(x) for x in gold_raw]
    else:
        gold_ids = [str(gold_raw)]

    return qid, qtext, gold_ids

def l2_normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.sqrt(np.maximum(np.sum(v * v, axis=axis, keepdims=True), eps))
    return v / denom

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = l2_normalize(a, axis=1)
    b = l2_normalize(b, axis=1)
    return a @ b.T

def main(k_max: int):
    print("[INFO] Loading JSON files...")
    with open(QUERIES_JSON, "r", encoding="utf-8") as f:
        queries_raw = json.load(f)

    with open(NAME_TO_DEF_JSON, "r", encoding="utf-8") as f:
        name_to_def = json.load(f)

    with open(ID_TO_NAME_JSON, "r", encoding="utf-8") as f:
        id_to_name = json.load(f)
    print("[INFO] Building statute catalogue (ID→canonical name→definition)...")
    statute_ids: List[str] = []
    statute_names: List[str] = []
    statute_texts: List[str] = []

    if not isinstance(id_to_name, dict):
        raise ValueError("id-section-meta.json must be a dict mapping slug -> [section, act, blurb, id].")

    missing_def = 0
    for _slug, arr in id_to_name.items():
        if not isinstance(arr, (list, tuple)) or len(arr) < 4:
            raise ValueError(f"Bad entry in id-section-meta.json for key '{_slug}': {arr}")

        section = str(arr[0]).strip()     
        act     = str(arr[1]).strip()    
        sid     = str(arr[3]).strip()    

        canonical_name = join_section_act(section, act)  
        definition = name_to_def.get(canonical_name, "").strip()
        if not definition:
            definition = canonical_name
            missing_def += 1

        statute_ids.append(sid)
        statute_names.append(canonical_name)
        statute_texts.append(definition)

    print(f"[INFO] Statutes: {len(statute_ids)} (missing definitions: {missing_def})")

    print("[INFO] Structuring queries...")
    q_ids, q_texts, q_golds = [], [], []
    for q in queries_raw:
        qid, qtext, gold_ids = extract_query(q)
        q_ids.append(qid)
        q_texts.append(qtext)
        q_golds.append([str(x) for x in gold_ids])
    N = len(q_ids)
    print(f"[INFO] Queries: {N}")

    print(f"[INFO] Loading SBERT: {MODEL_NAME} on {DEVICE}")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    print("[INFO] Encoding statutes...")
    statute_embs_parts: List[np.ndarray] = []
    for i in tqdm(range(0, len(statute_texts), BATCH_SIZE_STATUTES)):
        batch = statute_texts[i:i+BATCH_SIZE_STATUTES]
        embs = model.encode(
            batch,
            batch_size=BATCH_SIZE_STATUTES,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False
        )
        statute_embs_parts.append(embs)
    statute_embs = np.vstack(statute_embs_parts)  

    print("[INFO] Encoding queries...")
    query_embs_parts: List[np.ndarray] = []
    for i in tqdm(range(0, N, BATCH_SIZE_QUERIES)):
        batch = q_texts[i:i+BATCH_SIZE_QUERIES]
        embs = model.encode(
            batch,
            batch_size=BATCH_SIZE_QUERIES,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False
        )
        query_embs_parts.append(embs)
    query_embs = np.vstack(query_embs_parts)    

    print("[INFO] Computing similarities & rankings...")
    sims = cosine_sim(query_embs, statute_embs)        
    ranked_idx = np.argsort(-sims, axis=1)             

    print(f"[INFO] Writing rankings to {OUTPUT_RANKINGS_JSONL} (flush every 10 queries)...")
    ensure_dir(OUTPUT_RANKINGS_JSONL)
    open(OUTPUT_RANKINGS_JSONL, "w", encoding="utf-8").close()

    buffer: List[Dict[str, Any]] = []
    for i in range(N):
        order = ranked_idx[i].tolist()
        ids_ordered = [statute_ids[j] for j in order]
        scores_ordered = [float(sims[i, j]) for j in order]

        rec = {
            "query_id": q_ids[i],
            "gold_ids": q_golds[i],
            "top_statute_ids": ids_ordered,
            "top_scores": scores_ordered
        }
        buffer.append(rec)

        if (i + 1) % 10 == 0:
            with open(OUTPUT_RANKINGS_JSONL, "a", encoding="utf-8") as fout:
                for r in buffer:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            buffer.clear()

    if buffer:
        with open(OUTPUT_RANKINGS_JSONL, "a", encoding="utf-8") as fout:
            for r in buffer:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        buffer.clear()

    print("[INFO] Evaluating (macro & micro) for k = 1..{}...".format(k_max))
    gold_sets = [set(g) for g in q_golds]
    macro_results = {"Precision": [], "Recall": [], "F1": []}
    micro_results = {"Precision": [], "Recall": [], "F1": []}

    for k in range(1, k_max + 1):
        p_list, r_list, f1_list = [], [], []
        TP = FP = FN = 0

        for i in range(N):
            pred_k = set([statute_ids[j] for j in ranked_idx[i, :k]])
            gold   = gold_sets[i]
            tp = len(pred_k & gold)
            fp = len(pred_k - gold)
            fn = len(gold - pred_k)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            p_list.append(prec)
            r_list.append(rec)
            f1_list.append(f1)

            TP += tp; FP += fp; FN += fn

        macro_results["Precision"].append(float(np.mean(p_list)) if p_list else 0.0)
        macro_results["Recall"].append(float(np.mean(r_list)) if r_list else 0.0)
        macro_results["F1"].append(float(np.mean(f1_list)) if f1_list else 0.0)

        micro_prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        micro_rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        micro_f1   = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
        micro_results["Precision"].append(micro_prec)
        micro_results["Recall"].append(micro_rec)
        micro_results["F1"].append(micro_f1)

    k_cols = [f"k={k}" for k in range(1, k_max + 1)]
    table = pd.DataFrame(index=["mP", "muP", "mR", "muR", "mF1", "muF1"], columns=k_cols)

    table.loc["mP"]   = macro_results["Precision"]
    table.loc["muP"]  = micro_results["Precision"]
    table.loc["mR"]   = macro_results["Recall"]
    table.loc["muR"]  = micro_results["Recall"]
    table.loc["mF1"]  = macro_results["F1"]
    table.loc["muF1"] = micro_results["F1"]

    print("\n===== K-wise Metrics (rows: mP, muP, mR, muR, mF1, muF1; cols: k=1..k) =====")
    print(table.to_string(float_format=lambda x: f"{float(x):.4f}"))

    ensure_dir(OUTPUT_K_TABLE_CSV)
    table.to_csv(OUTPUT_K_TABLE_CSV)

    summary = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "num_queries": N,
        "num_statutes": len(statute_ids),
        "k_max": k_max,
        "metrics_table_csv": OUTPUT_K_TABLE_CSV,
        "paths": {
            "rankings_jsonl": OUTPUT_RANKINGS_JSONL,
            "summary_json": OUTPUT_SUMMARY_JSON
        }
    }
    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Saved:")
    print(f" - Rankings JSONL: {OUTPUT_RANKINGS_JSONL}")
    print(f" - K-table CSV   : {OUTPUT_K_TABLE_CSV}")
    print(f" - Summary JSON  : {OUTPUT_SUMMARY_JSON}")
    print("[DONE]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-max", type=int, default=K_MAX_DEFAULT, help="Evaluate from k=1..k_max (default 10)")
    args = parser.parse_args()
    main(k_max=args.k_max)