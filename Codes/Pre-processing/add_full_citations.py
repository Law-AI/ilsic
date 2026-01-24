import json


REFERENCE_FILE = r"/home/shounak/HDD/Layman-LSI/dataset/id-section-meta.json"
INPUT_FILE = r"/home/shounak/HDD/Layman-LSI/dataset/court-dataset/id-court-train-final-clean.json"
OUTPUT_FILE = r"/home/shounak/HDD/Layman-LSI/dataset/court-dataset/id-court-train-final-clean.json"


with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
    reference = json.load(f)


citation_lookup = {}
for ref in reference.values():
    citation_id = ref[3]
    citation_lookup[citation_id] = f"{ref[0]} of {ref[1]}"


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    input_data = json.load(f)


for obj in input_data:
    full_citations = []
    for cid in obj.get("citations-id", []):
        if cid in citation_lookup:
            full_citations.append(citation_lookup[cid])
    obj["full-citations"] = full_citations


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(input_data, f, indent=2, ensure_ascii=False)

print(f"Processed {len(input_data)} entries. Output written to {OUTPUT_FILE}")
