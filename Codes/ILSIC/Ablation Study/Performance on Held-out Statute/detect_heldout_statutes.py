"""
Detect held-out statutes in test split and save separate lists of held-out statute IDs and test query IDs.

Outputs:
  out/held_out_report.json
  out/held_out_statute_ids.json
  out/held_out_query_ids.json
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Set


TRAIN_PATH = "/home/shounak/HDD/Layman-LSI/dataset/Layman-new-dataset/id-Layman-train-clean.json"
DEV_PATH   = "/home/shounak/HDD/Layman-LSI/dataset/Layman-new-dataset/id-Layman-dev-clean.json"
TEST_PATH  = "/home/shounak/HDD/Layman-LSI/dataset/Layman-new-dataset/id-Layman-test-clean.json"

OUT_DIR = Path("/home/shounak/HDD/Layman-LSI/dataset/Layman-new-dataset/heldout-statutes")
OUT_DIR.mkdir(parents=True, exist_ok=True)



def load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), f"Expected a list in {path}"
    return data


def extract_statute_ids(rows: List[Dict[str, Any]]) -> Set[str]:
    ids = set()
    for obj in rows:
        for sid in obj.get("citations-id", []) or []:
            if sid is not None:
                ids.add(str(sid))
    return ids


def main():
    # Load data
    train_rows = load_json_array(TRAIN_PATH)
    dev_rows   = load_json_array(DEV_PATH)
    test_rows  = load_json_array(TEST_PATH)

    train_ids = extract_statute_ids(train_rows)
    dev_ids   = extract_statute_ids(dev_rows)
    test_ids  = extract_statute_ids(test_rows)

    seen_train_dev = train_ids | dev_ids
    held_out_ids = test_ids - seen_train_dev

    held_map: Dict[str, List[int]] = {sid: [] for sid in held_out_ids}
    test_query_ids_any: Set[int] = set()

    for obj in test_rows:
        qid = obj.get("id")
        if qid is None:
            continue
        cited = {str(c) for c in (obj.get("citations-id", []) or [])}
        inter = cited & held_out_ids
        if inter:
            test_query_ids_any.add(int(qid))
            for sid in inter:
                held_map[sid].append(int(qid))

    held_out_statute_ids = sorted(list(held_out_ids), key=lambda x: (len(x), x))
    held_out_query_ids   = sorted(list(test_query_ids_any))

    report = {
        "held_out_statute_count": len(held_out_statute_ids),
        "held_out_statute_ids": held_out_statute_ids,
        "held_out_query_ids": held_out_query_ids,
        "held_out_statute_to_test_query_ids": held_map,
        "summary": {
            "train_unique_statutes": len(train_ids),
            "dev_unique_statutes": len(dev_ids),
            "test_unique_statutes": len(test_ids),
            "seen_in_train_or_dev": len(seen_train_dev)
        }
    }

    (OUT_DIR / "held_out_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "held_out_statute_ids.json").write_text(json.dumps(held_out_statute_ids, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "held_out_query_ids.json").write_text(json.dumps(held_out_query_ids, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] Held-out statutes: {len(held_out_statute_ids)}")
    print(f"[DONE] Test queries with held-out statutes: {len(held_out_query_ids)}")
    print(f"Files saved in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
