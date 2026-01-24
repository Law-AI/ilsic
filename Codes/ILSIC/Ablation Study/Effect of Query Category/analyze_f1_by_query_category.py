import json
from collections import defaultdict
from sklearn.metrics import f1_score
import numpy as np


TEST_DATA_PATH = '/home/shounak/HDD/Layman-LSI/dataset/Layman-subset-data/id-LMsub-test-clean.json'
PREDICTIONS_PATH = '/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/two-shot/id-base_gpt4-1_two_shot.json'
OUTPUT_PATH = '/home/shounak/HDD/Layman-LSI/Rebuttal_ARR_July/analysis/category-wise/gpt_LMnew_category_analysis.json'


with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
with open(PREDICTIONS_PATH, 'r', encoding='utf-8') as f:
    pred_data = json.load(f)


test_by_id = {entry['id']: entry for entry in test_data}
pred_by_id = {entry['query_no']: entry for entry in pred_data}


category_to_queries = defaultdict(list)
for entry in test_data:
    category = entry['query-category']
    category_to_queries[category].append(entry['id'])


def get_f1(true_ids, pred_ids):
    y_true = [str(x) for x in true_ids]
    y_pred = [str(x) for x in pred_ids]
    all_labels = list(set(y_true) | set(y_pred))
    y_true_bin = [1 if lbl in y_true else 0 for lbl in all_labels]
    y_pred_bin = [1 if lbl in y_pred else 0 for lbl in all_labels]
    if sum(y_true_bin) == 0 and sum(y_pred_bin) == 0:
        return 1.0
    return f1_score(y_true_bin, y_pred_bin)

query_f1 = {}
for entry in test_data:
    qid = entry['id']
    true_ids = entry.get('citations-id', [])
    pred_entry = pred_by_id.get(qid)
    if pred_entry:
        pred_ids = pred_entry.get('matched_ids', [])
    else:
        pred_ids = []
    query_f1[qid] = get_f1(true_ids, pred_ids)


cat_f1 = {}
for cat, qids in category_to_queries.items():
    f1s = [query_f1[qid] for qid in qids if qid in query_f1]
    if f1s:
        cat_f1[cat] = np.mean(f1s)
    else:
        cat_f1[cat] = None


print("Average F1-score by category:")
for cat, f1_val in cat_f1.items():
    print(f"{cat}: {f1_val:.3f}" if f1_val is not None else f"{cat}: No data")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(cat_f1, f, indent=2)
