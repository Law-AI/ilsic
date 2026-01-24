import json
from collections import Counter, OrderedDict
import numpy as np
from sklearn.metrics import f1_score


test_path = "/home/shounak/HDD/Layman-LSI/dataset/Layman-new-dataset/id-Layman-test-clean.json"
model_output_path = "/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/two-shot/id-base_gpt4-1_two_shot.json"
output_json_path = "/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/analysis/freq/gpt_LMnew_Freq_10grp_f1_μ_AllQ.json"
NUM_GROUPS = 10
FILTER_NON_ZERO = False 



def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_statute_ids(data):
    all_ids = []
    for entry in data:
        all_ids.extend(map(str, entry.get("citations-id", [])))
    return all_ids

def get_test_labels_and_preds(test_data, pred_data):
    id_to_true = {str(entry["id"]): list(map(str, entry.get("citations-id", []))) for entry in test_data}
    id_to_pred = {str(entry["query_no"]): list(map(str, entry.get("matched_ids", []))) for entry in pred_data}
    all_ids = sorted(id_to_true.keys(), key=lambda x: int(x))
    y_true = [id_to_true[k] for k in all_ids]
    y_pred = [id_to_pred.get(k, []) for k in all_ids]
    return all_ids, y_true, y_pred

def build_statute_index(statute_ids):
    uniq_ids = sorted(set(statute_ids), key=lambda x: int(x))
    return uniq_ids, {sid: i for i, sid in enumerate(uniq_ids)}

def multilabel_matrix(y_list, statute_to_idx, num_statutes):
    mat = np.zeros((len(y_list), num_statutes), dtype=int)
    for i, label_ids in enumerate(y_list):
        for sid in label_ids:
            if sid in statute_to_idx:
                mat[i, statute_to_idx[sid]] = 1
    return mat



def main():
    test = load_json(test_path)
    pred = load_json(model_output_path)

    all_statute_ids = extract_statute_ids(test)
    freq_counter = Counter(all_statute_ids)
    ranked_statutes = [sid for sid, _ in freq_counter.most_common()]  

    total = len(ranked_statutes)
    group_size = (total + NUM_GROUPS - 1) // NUM_GROUPS  
    statute_groups = [ranked_statutes[i*group_size:(i+1)*group_size] for i in range(NUM_GROUPS)]

    test_ids, y_true_list, y_pred_list = get_test_labels_and_preds(test, pred)
    uniq_statutes, statute_to_idx = build_statute_index(all_statute_ids)
    num_statutes = len(uniq_statutes)

    y_true_mat = multilabel_matrix(y_true_list, statute_to_idx, num_statutes)
    y_pred_mat = multilabel_matrix(y_pred_list, statute_to_idx, num_statutes)

    f1_macro = f1_score(y_true_mat, y_pred_mat, average='macro', zero_division=0)
    f1_micro = f1_score(y_true_mat, y_pred_mat, average='micro', zero_division=0)
    
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (micro): {f1_micro:.4f}")
    
    results = []

    for i, group in enumerate(statute_groups):
        group_freqs = {sid: freq_counter[sid] for sid in group}
        group_freqs_ordered = OrderedDict(
            sorted(group_freqs.items(), key=lambda x: (-x[1], int(x[0])))
        )
        selected_cols = [statute_to_idx[sid] for sid in group if sid in statute_to_idx]

        if not selected_cols:
            f1 = None
        else:
            y_true_mat = multilabel_matrix(y_true_list, statute_to_idx, num_statutes)[:, selected_cols]
            y_pred_mat = multilabel_matrix(y_pred_list, statute_to_idx, num_statutes)[:, selected_cols]

            num_all_rows = y_true_mat.shape[0]
            if FILTER_NON_ZERO:
                mask = (y_true_mat.sum(axis=1) > 0) | (y_pred_mat.sum(axis=1) > 0)
                num_included = mask.sum()
                print(f"Group {i+1}: {num_included} of {num_all_rows} rows included after non-zero filtering.")
                y_true_mat = y_true_mat[mask]
                y_pred_mat = y_pred_mat[mask]
            else:
                print(f"Group {i+1}: All {num_all_rows} rows included (no filtering).")

            if y_true_mat.shape[0] == 0:
                f1 = None
            else:
                f1 = f1_score(y_true_mat, y_pred_mat, average='micro', zero_division=0)

        group_statute_ids_str = json.dumps(group_freqs_ordered, separators=(",", ":"))

        results.append({
            "group_index": i + 1,
            "group_frequency_rank": i + 1,
            "group_statute_ids": group_statute_ids_str,
            "f1_micro": f1
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved analysis to {output_json_path}")

if __name__ == "__main__":
    main()
