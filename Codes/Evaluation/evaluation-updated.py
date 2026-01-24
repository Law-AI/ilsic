# ===================================================================
# HOW TO USE THIS FILTERING SYSTEM
#
# You can control filtering for ROWS (queries) and COLUMNS (statutes)
# independently using dictionaries:
#
#   {"mode": "none"}
#       → Keep everything (no filtering).
#
#   {"mode": "positions", "path": "file.json", "key": "test", "one_based": True}
#       → Select by row/column positions (not IDs).
#         - File should be either [1,2,3,...] or {"test": [1,2,3,...]}
#         - If "one_based": True → converts 1-based indices to 0-based
#
#   {"mode": "ids", "path": "file.json", "key": "test"}
#       → Select by actual IDs.
#         - File should be either [420,415,...] or {"test":[420,415,...]}
#         - For rows: IDs are query IDs
#         - For columns: IDs are statute IDs
#
# Examples:
#   row_filter = {"mode":"none"}  → all queries kept
#   col_filter = {"mode":"none"}  → all statutes kept
#
#   row_filter = {"mode":"positions", "path":"queries.json", "key":"test", "one_based":True}
#       → use positions listed in queries.json["test"]
#
#   row_filter = {"mode":"ids", "path":"queries.json", "key":"test"}
#       → use actual query IDs listed in queries.json["test"]
#
#   col_filter = {"mode":"ids", "path":"statutes.json"}
#       → use statute IDs listed in statutes.json
#
# ===================================================================

import json
import numpy as np
import re
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
import csv
from typing import Dict, List, Tuple, Optional

def extract_ids(citations):
    """Extract integers from citations-id or matched_ids."""
    ids = []
    for elem in citations:
        for sid in re.findall(r'\d+', str(elem)):
            try:
                ids.append(int(sid))
            except ValueError:
                continue
    return ids

def build_statute_ids(statutes_path: str) -> Tuple[List[int], Dict[int, int]]:
    """Extract canonical statute IDs from arr[3]."""
    with open(statutes_path, 'r', encoding='utf-8') as f:
        statutes = json.load(f)

    statute_ids = []
    for arr in statutes.values():
        if len(arr) >= 4:
            try:
                sid = int(arr[3])  
                statute_ids.append(sid)
            except (ValueError, IndexError):
                continue

    statute_ids = sorted(set(statute_ids))
    statute_id_to_idx = {sid: idx for idx, sid in enumerate(statute_ids)}

    print("\n=== STATUTE SPACE DEBUG ===")
    print("Total statute_ids:", len(statute_ids))
    print("First 20 statute_ids:", statute_ids[:20])
    print("Last 20 statute_ids:", statute_ids[-20:], "\n")

    return statute_ids, statute_id_to_idx


def build_matrix(json_path, id_key, citations_key, statute_id_to_idx, num_statutes):
    with open(json_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    matrix_dict = {}

    for item in items:
        qid = item.get(id_key)
        if qid is None:
            continue

        vector = [0] * num_statutes
        ids = extract_ids(item.get(citations_key, []))

        for cid in ids:
            if cid in statute_id_to_idx:
                vector[statute_id_to_idx[cid]] = 1

        matrix_dict[int(qid)] = vector

    return matrix_dict

def align_and_build_matrix(dict_matrix, statute_ids, all_query_ids):
    num_statutes = len(statute_ids)
    matrix = []
    for qid in all_query_ids:
        if qid in dict_matrix:
            matrix.append(dict_matrix[qid])
        else:
            matrix.append([0] * num_statutes)
    return np.array(matrix, dtype=int)


def _load_vector_from_json(path: str, key: Optional[str]):
    with open(path, "r", encoding="utf-8") as fr:
        obj = json.load(fr)
    if key is None:
        return [int(x) for x in obj]
    else:
        return [int(x) for x in obj[key]]

def _resolve_row_indices(row_filter: Dict, all_query_ids: List[int]):
    mode = row_filter.get("mode", "none")

    if mode == "none":
        return np.arange(len(all_query_ids))

    if mode == "positions":
        pos = _load_vector_from_json(row_filter["path"], row_filter.get("key"))
        pos = np.array(pos) - 1
        pos = pos[(pos >= 0) & (pos < len(all_query_ids))]
        return pos

    if mode == "ids":
        ids = _load_vector_from_json(row_filter["path"], row_filter.get("key"))
        mapping = {qid: i for i, qid in enumerate(all_query_ids)}
        idx = [mapping[q] for q in ids if q in mapping]
        return np.array(idx)

    raise ValueError("Unknown row_filter mode")

def _resolve_col_indices(col_filter: Dict, statute_ids: List[int]):
    mode = col_filter.get("mode", "none")

    if mode == "none":
        return np.arange(len(statute_ids))

    if mode == "positions":
        pos = _load_vector_from_json(col_filter["path"], col_filter.get("key"))
        pos = np.array(pos) - 1
        pos = pos[(pos >= 0) & (pos < len(statute_ids))]
        return pos

    if mode == "ids":
        ids = _load_vector_from_json(col_filter["path"], col_filter.get("key"))
        mapping = {sid: i for i, sid in enumerate(statute_ids)}
        idx = [mapping[s] for s in ids if s in mapping]
        return np.array(idx)

    raise ValueError("Unknown col_filter mode")


def main(statutes_path, queries_path, llm_outputs_path, results_json_path,
         row_filter=None, col_filter=None, save_matrices_csv=0,
         gt_csv_path="gt_matrix.csv", resp_csv_path="resp_matrix.csv"):

    row_filter = row_filter or {"mode": "none"}
    col_filter = col_filter or {"mode": "none"}

    
    statute_ids, statute_id_to_idx = build_statute_ids(statutes_path)
    num_statutes = len(statute_ids)

    
    gt_dict = build_matrix(queries_path, 'id', 'citations-id', statute_id_to_idx, num_statutes)
    # resp_dict = build_matrix(llm_outputs_path, 'query_no', 'matched_ids', statute_id_to_idx, num_statutes) ##<--- Original
    resp_dict = build_matrix(llm_outputs_path, 'query-id', 'matched_ids', statute_id_to_idx, num_statutes)

    all_query_ids = sorted(set(gt_dict.keys()).union(resp_dict.keys()))

    gt_matrix = align_and_build_matrix(gt_dict, statute_ids, all_query_ids)
    resp_matrix = align_and_build_matrix(resp_dict, statute_ids, all_query_ids)

    print("\n=== MATRIX SHAPES ===")
    print("GT matrix:", gt_matrix.shape)
    print("Resp matrix:", resp_matrix.shape)

    row_idx = _resolve_row_indices(row_filter, all_query_ids)
    col_idx = _resolve_col_indices(col_filter, statute_ids)

    gt_matrix_f = gt_matrix[row_idx][:, col_idx]
    resp_matrix_f = resp_matrix[row_idx][:, col_idx]

    print("\n=== FILTERED MATRIX SHAPES ===")
    print("GT matrix filtered:", gt_matrix_f.shape)
    print("Resp matrix filtered:", resp_matrix_f.shape)

    gt_sum = gt_matrix_f.sum(axis=0)
    resp_sum = resp_matrix_f.sum(axis=0)

    active_gt = np.where(gt_sum > 0)[0]
    active_pred = np.where(resp_sum > 0)[0]
    active_both = np.intersect1d(active_gt, active_pred)

    print("\n=== STATUTE ACTIVITY DEBUG ===")
    print("Total statutes after filtering:", len(col_idx))
    print("Statutes with GT positives:", len(active_gt))
    print("Statutes with Pred positives:", len(active_pred))
    print("Statutes active in BOTH:", len(active_both))
    print("Statute IDs active in both:", [statute_ids[col_idx[i]] for i in active_both], "\n")

    precision_macro = precision_score(gt_matrix_f, resp_matrix_f, average='macro', zero_division=0)
    precision_micro = precision_score(gt_matrix_f, resp_matrix_f, average='micro', zero_division=0)
    recall_macro = recall_score(gt_matrix_f, resp_matrix_f, average='macro', zero_division=0)
    recall_micro = recall_score(gt_matrix_f, resp_matrix_f, average='micro', zero_division=0)
    f1_macro = f1_score(gt_matrix_f, resp_matrix_f, average='macro', zero_division=0)
    f1_micro = f1_score(gt_matrix_f, resp_matrix_f, average='micro', zero_division=0)

    print("\n=== PER-LABEL P/R/F1 DEBUG ===")
    p_arr, r_arr, f_arr, support_arr = precision_recall_fscore_support(
        gt_matrix_f, resp_matrix_f, average=None, zero_division=0
    )

    for i, (p, r, f, s) in enumerate(zip(p_arr, r_arr, f_arr, support_arr)):
        if p != 0 or r != 0 or f != 0:
            sid = statute_ids[col_idx[i]]
            print(f"Label idx={i} (statute {sid}): P={p:.4f}, R={r:.4f}, F1={f:.4f}, support={s}")

    gt_ids_set = set(gt_dict.keys())
    model_ids_set = set(resp_dict.keys())

    model_ids_missing_in_gt = sorted(list(model_ids_set - gt_ids_set))
    gt_ids_missing_in_model = sorted(list(gt_ids_set - model_ids_set))

    filtered_query_ids = [all_query_ids[i] for i in row_idx]

    alignment_warning = ""
    if len(model_ids_missing_in_gt) > 0:
        alignment_warning = (
            "WARNING: Model output contains query IDs that DO NOT EXIST in GT. "
            "These rows will become all-zero predictions after filtering."
        )
    elif len(gt_ids_missing_in_model) > 0:
        alignment_warning = (
            "WARNING: GT contains query IDs that model did NOT predict. "
            "These rows will have zero predictions."
        )
    else:
        alignment_warning = "GT and Model query-id alignment OK."

    results = {
        "statutes_path": statutes_path,
        "queries_path": queries_path,
        "llm_outputs_path": llm_outputs_path,

        "precision_macro": precision_macro,
        "precision_micro": precision_micro,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        
        "row_filter": row_filter,
        "col_filter": col_filter,

        "kept_num_queries": len(row_idx),
        "kept_num_statutes": len(col_idx),


        "all_query_ids": all_query_ids,
        "filtered_query_ids": filtered_query_ids,
        "model_query_ids": sorted(list(model_ids_set)),
        "gt_query_ids": sorted(list(gt_ids_set)),
        "model_ids_missing_in_gt": model_ids_missing_in_gt,
        "gt_ids_missing_in_model": gt_ids_missing_in_model,
        "alignment_warning": alignment_warning

        

        
    }


    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL METRICS ===")
    print(f"Precision (macro): {precision_macro:.4f} | Precision (micro): {precision_micro:.4f}")
    print(f"Recall (macro):    {recall_macro:.4f} | Recall (micro):    {recall_micro:.4f}")
    print(f"F1 (macro):        {f1_macro:.4f} | F1 (micro):        {f1_micro:.4f}")
    print(f"Queries kept: {len(row_idx)}, Statutes kept: {len(col_idx)}\n")



if __name__ == "__main__":
    statutes_path = r"/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/id-section-meta.json"
    queries_gt_path = r"/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/Layman-new-dataset/id-Layman-test-clean.json"
    llm_outputs_path = r"/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/Reasoning-Experiment/inf-result/id-two-shot-FT-llama-54q-expert-explanation-exp.json"
    results_json_path = r"/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/Reasoning-Experiment/eval-result/two-shot_FT_llama_54q_reasoning-exp.json"

    ### Check this once for what you are using-> resp_dict

    # --- EXAMPLES: pick what you need ---
    # 1) No filtering (use all rows/columns)
    # row_filter = {"mode": "none"}
    # col_filter = {"mode": "none"}

    # 2) Filter rows by 1-based positions from LMsubset_query_id_list.json, key "test"
    ### Row = Queries
    row_filter = {
        # "mode": "none"
        "mode": "ids",
        "path": r"/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/queries-id-list-to-annotate.json",
        "key": None,      # if the JSON is {"test":[...]}
        # "one_based": False   # subtract 1 internally
    }

    # 3) Filter columns by IDs from synLM_cits_idx_list.json (assuming it stores statute IDs)
    ### Column = Statutes
    col_filter = {
        "mode": "none"
        # "mode": "ids",
        # "path": r"/home/shounak/HDD/Layman-LSI/dataset/Layman-new-dataset/heldout-statutes/held_out_statute_ids.json",
        # "key": None       # omit key if the JSON is a plain list
    }

    # 4) Alternative: filter columns by positions (1-based) if your JSON is positions
    # col_filter = {
    #     "mode": "positions",
    #     "path": r"/path/to/positions.json",
    #     "key": "test",
    #     "one_based": True
    # }

    save_matrices_csv = 0
    main(statutes_path, queries_gt_path, llm_outputs_path, results_json_path,
         row_filter=row_filter, col_filter=col_filter,
         save_matrices_csv=save_matrices_csv)

