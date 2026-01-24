import json

JSON1_PATH = "/home/shounak/HDD/Layman-LSI/dataset/section-texts.json"   
JSON2_PATH = "/home/shounak/HDD/Layman-LSI/dataset/id-section-meta.json"   
OUTPUT_PATH = "/home/shounak/HDD/Layman-LSI/dataset/section-formal-texts.json"  
UNMATCHED_REPORT_PATH = "/home/shounak/HDD/Layman-LSI/dataset/unmatched_keys.txt"

def normalize_key(k: str) -> str:
    return " ".join(k.strip().lower().split())

def main():
    with open(JSON1_PATH, "r", encoding="utf-8") as f:
        j1 = json.load(f)
    with open(JSON2_PATH, "r", encoding="utf-8") as f:
        j2 = json.load(f)

    j2_lookup = {normalize_key(k): v for k, v in j2.items()}

    result = {}
    unmatched = []

    for raw_key, long_text in j1.items():
        key_norm = normalize_key(raw_key)
        meta = j2_lookup.get(key_norm)

        if meta and isinstance(meta, list) and len(meta) >= 2:
            section = meta[0]         
            act = meta[1]             
            new_key = f"{section} of {act}"
            result[new_key] = long_text
        else:
            result[raw_key] = long_text
            unmatched.append(raw_key)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if unmatched:
        with open(UNMATCHED_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("Keys from JSON 1 with no match in JSON 2:\n\n")
            for k in unmatched:
                f.write(k + "\n")

    print(f"Done. Wrote renamed JSON to: {OUTPUT_PATH}")
    if unmatched:
        print(f"{len(unmatched)} keys were not matched. See: {UNMATCHED_REPORT_PATH}")
    else:
        print("All keys matched.")

if __name__ == "__main__":
    main()
