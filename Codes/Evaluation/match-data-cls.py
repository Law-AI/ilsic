import sys
import json
import os
import re
from rapidfuzz import fuzz
from typing import List, Tuple, Dict, Optional

CONNECTIVE_FILLERS = {
    "under", "of", "by", "on", "in", "at", "with", "and", "for", "from", "to",
    "a", "an", "or", "but", "as", "per", "via", "upon", "within", "into",
    "through", "over", "between", "among", "around", "about", "before", "after",
    "during", "since", "until", "towards", "against", "without", "including",
    "regarding", "concerning", "according", "versus", "vs", "w.r.t",
}

def remove_connective_fillers(text: str) -> str:
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

def normalize_identifier_token(token: str) -> str:
    result = []
    for c in token:
        if c in "([":  
            break
        if c.isalnum():
            result.append(c)
    return "".join(result)

def analyze_text_structure(text: str) -> Dict:
    if not text:
        return {}
    words = text.split()
    analysis = {
        "words": words,
        "numeric_words": [],
        "short_alphanum_words": [],
        "potential_identifiers": [],
    }
    for i, word in enumerate(words):
        normalized = normalize_identifier_token(word)
        if normalized.isdigit():
            analysis["numeric_words"].append((i, normalized))
        elif len(normalized) <= 5 and any(c.isdigit() for c in normalized):
            analysis["short_alphanum_words"].append((i, normalized))
        if len(word) <= 6 and (
            any(c.isdigit() for c in normalized) or any(x in word for x in "()[].")
        ):
            analysis["potential_identifiers"].append((i, word))
    return analysis

def find_section_identifier(text: str) -> Tuple[Optional[str], int]:
    analysis = analyze_text_structure(text)
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
            cleaned = normalize_identifier_token(word)
            if cleaned:
                return cleaned, pos

    return None, -1

def intelligent_split_section_act(text: str) -> Tuple[Optional[str], Optional[str]]:
    text = text.strip()
    if not text:
        return None, None

    words = text.split()
    identifier, pos = find_section_identifier(text)

    if identifier and pos >= 0:
        section_words = []
        if pos > 0:
            prev = words[pos - 1]
            if not any(c.isdigit() for c in prev):
                section_words.append(prev)
        section_words.append(identifier)

        section = " ".join(section_words).strip()
        used = set(w.lower() for w in section_words)
        act = " ".join(w for w in words if w.lower() not in used).strip()
        return section, act

    return None, text

def extract_section_act_pairs_intelligent(output_text):
    pairs_list = []

    if isinstance(output_text, str):
        clean = output_text.strip().strip("[]")
        clean = re.sub(r'^"+|"+$', '', clean)
        splits = [s.strip().strip("\"'") for s in re.split(r"\s*;\s*", clean) if s.strip()]
        pairs_list.extend(splits)

    elif isinstance(output_text, list):
        for item in output_text:
            pairs_list.append(item.strip())

    extracted = []
    for pair in pairs_list:
        sec, act = intelligent_split_section_act(pair)
        if sec:
            extracted.append((sec, act if act else ""))

    return extracted

def normalize_for_comparison(text: str) -> str:
    if not text:
        return ""
    text = remove_connective_fillers(text)
    return " ".join("".join(c for c in w if c.isalnum()) for w in text.split())

def smart_fuzzy_match(t1: str, t2: str) -> float:
    if not t1 or not t2:
        return 0.0

    scores = [
        fuzz.ratio(t1, t2),
        fuzz.token_sort_ratio(t1, t2),
        fuzz.token_set_ratio(t1, t2),
        fuzz.ratio(normalize_for_comparison(t1), normalize_for_comparison(t2)),
    ]
    return max(scores)

def get_matched_ids(
    pairs: List[Tuple[str, str]],
    reference_data: Dict[str, List[str]],
    threshold: int
) -> List[str]:

    output_ids = []

    for section_input, act_input in pairs:
        input_id, _ = find_section_identifier(section_input)
        best_score = 0.0
        best_id = None

        for ref in reference_data.values():
            if len(ref) < 4:
                continue

            ref_section, ref_act, _, ref_id = ref
            ref_identifier, _ = find_section_identifier(ref_section)

            if not input_id or not ref_identifier:
                continue

            if input_id.lower() != ref_identifier.lower():
                continue

            score = smart_fuzzy_match(act_input.lower(), ref_act.lower())
            if score > best_score:
                best_score = score
                best_id = ref_id

        output_ids.append(best_id if best_score >= threshold and best_id else "NA")

    return output_ids

def resolve_annotated_ids(
    annotated_ids: List[str],
    reference_data: Dict[str, List[str]]
) -> List[str]:

    resolved = []
    for ref in reference_data.values():
        if len(ref) < 4:
            continue
        section, act, _, ref_id = ref
        if ref_id in annotated_ids:
            resolved.append(f"{section} of {act}")
    return resolved

if __name__ == "__main__":

    input_json_path = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/extended-data/gpt5_2_pred-new-small-train-4.json"
    reference_json_path = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/id-section-meta.json"
    threshold = 50

    # mode:
    # 0 → citations → citations-id
    # 1 → model_output → matched_ids
    # 2 → annotated_ids → annotated_statutes
    # =============================

    MODE = 1   

    mode = MODE


    if mode == 0:
        key_name = "citations"
        output_key = "citations-id"
    elif mode == 1:
        key_name = "model_output"
        output_key = "matched_ids"
    elif mode == 2:
        key_name = "annotated_ids"
        output_key = "annotated_full_citations"

    else:
        raise ValueError("Invalid mode")

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(reference_json_path, "r", encoding="utf-8") as f:
        reference_data = json.load(f)

    for obj in data:
        key_value = obj.get(key_name, [])

        if mode in (0, 1):
            pairs = extract_section_act_pairs_intelligent(key_value)
            obj[output_key] = get_matched_ids(pairs, reference_data, threshold)

        elif mode == 2:
            obj[output_key] = resolve_annotated_ids(key_value, reference_data)

        if mode == 1 and "id" in obj:
            obj["query_no"] = obj.pop("id")

    out_path = os.path.join(
        os.path.dirname(input_json_path),
        f"id-{os.path.basename(input_json_path)}"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Processed successfully → {out_path}")
