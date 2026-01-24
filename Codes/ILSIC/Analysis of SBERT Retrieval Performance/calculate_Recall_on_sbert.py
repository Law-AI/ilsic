import json

# ----------------------------
# Load both json files
# ----------------------------

# LLM output JSON (SBERT top-15)
LLM_OUTPUT = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/Rebuttal_ARR_July/RAG/restricted/QQ-sbert/id-gemma_base_RAG_kStatfrom_QQ.json"
with open(LLM_OUTPUT, "r", encoding="utf-8") as f:
    llm_output = json.load(f)

# Ground-truth JSON (test set)
GT_JSON = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/Layman-new-dataset/id-Layman-test-clean-org.json"
with open(GT_JSON, "r", encoding="utf-8") as f:
    test_data = json.load(f)


# ------------------------------------------------------------
# Compute Recall@15 per query and final average Recall@15
# ------------------------------------------------------------
recalls = []

for llm_entry, test_entry in zip(llm_output, test_data):

    # SBERT predictions (top 15)
    preds = set(llm_entry["candidate_ids"])

    # Ground truth (gold)
    gold = set(test_entry["citations-id"])

    # If gold is empty (rare), skip
    if len(gold) == 0:
        continue

    # Intersection count
    correct = len(preds.intersection(gold))

    # Recall@15
    recall = correct / len(gold)
    recalls.append(recall)

    print(f"Query {test_entry['id']}: Recall@15 = {recall:.4f}  "
          f"(matched {correct}/{len(gold)})")


# ------------------------------------------------------------
# Final Macro Recall@15
# ------------------------------------------------------------
average_recall = sum(recalls) / len(recalls)

print("\n====================================")
print(f"Final Recall@15 = {average_recall:.4f}")
print("====================================")
