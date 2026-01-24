#!/usr/bin/env python3
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
        "word_count": len(words),
        "numeric_words": [],
        "short_alphanum_words": [],
        "long_words": [],
        "potential_identifiers": [],
    }
    for i, word in enumerate(words):
        normalized = normalize_identifier_token(word)
        if normalized.isdigit():
            analysis["numeric_words"].append((i, normalized))
        elif len(normalized) <= 5 and any(c.isdigit() for c in normalized):
            analysis["short_alphanum_words"].append((i, normalized))
        elif len(normalized) > 8:
            analysis["long_words"].append((i, normalized))
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
    if len(words) < 2:
        return text, ""
    identifier, pos = find_section_identifier(text)
    if identifier and pos >= 0:
        section_words = []
        if pos > 0:
            prev_word = words[pos - 1]
            if len(prev_word) <= 10 and not any(c.isdigit() for c in prev_word):
                section_words.append(prev_word)
        section_words.append(identifier)
        section = " ".join(section_words).strip()
        used = set(w.lower() for w in section_words)
        remaining = [w for w in words if w.lower() not in used]
        act = " ".join(remaining).strip()
        return section, act
    return None, text

def parse_raw_generation_flexible(raw: str) -> List[str]:
    """
    Parse raw_generation that may be:
      - a JSON array string with commas (e.g., "[\"Section 125 ...\", \"Section 13 ...\"]")
      - a semicolon-separated pseudo-list (e.g., "["A"; "B"; "C"]")
      - a quoted, comma-separated pseudo-list (not valid JSON)
    Returns list[str].
    """
    if not isinstance(raw, str):
        return []

    s = raw.strip()

    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    

    if s.startswith("[") and s.endswith("]"):
        s_inner = s[1:-1].strip()
    else:
        s_inner = s

    if ";" in s_inner:
        parts = re.split(r"\s*;\s*", s_inner)
    else:
        parts = re.split(r'\s*,\s*(?=["\'])', s_inner)

    items = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
            p = p[1:-1]
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            items.append(p)
    return items

def extract_section_act_pairs_intelligent(output_text) -> List[Tuple[str, str]]:
    """
    Normalize to a list[str], then split cautiously on commas that start new 'Section' tokens
    or short numeric section refs, and finally map to (section, act).
    """
    if isinstance(output_text, str):
        raw_items = parse_raw_generation_flexible(output_text)
    elif isinstance(output_text, list):
        raw_items = [str(x).strip() for x in output_text if str(x).strip()]
    else:
        raise ValueError("Model output must be a string or a list of strings.")

    pairs_list: List[str] = []
    for item in raw_items:
        sub_splits = re.split(
            r"""\s*,\s*(?=(?:[Ss](?:ection|ec)\.?\s*\d))""",
            item
        )
        for p in sub_splits:
            s = p.strip().strip("\"'")
            if s:
                pairs_list.append(s)

    cleaned_pairs_list: List[str] = []
    for s in pairs_list:
        if re.fullmatch(r"\d{1,3}[A-Za-z]?", s.strip()):
            continue
        cleaned_pairs_list.append(s.strip())
    pairs_list = cleaned_pairs_list

    extracted: List[Tuple[str, str]] = []
    for pair in pairs_list:
        sec, act = intelligent_split_section_act(pair)
        if not sec:
            m = re.search(r"\b(\d{1,3}[A-Za-z]?)\b", pair)
            if m:
                sec = m.group(1)
                act = pair.replace(m.group(0), "", 1).strip()
        if sec:
            extracted.append((sec, act if act else ""))
    return extracted

def normalize_for_comparison(text: str) -> str:
    if not text:
        return ""
    text = remove_connective_fillers(text)
    words = text.split()
    filtered = []
    for word in words:
        clean_word = "".join(c for c in word if c.isalnum())
        if (
            any(c.isdigit() for c in clean_word)
            or len(clean_word) > 2
            or (len(clean_word) == 2 and clean_word.isalpha())
        ):
            filtered.append(clean_word)
    if len(filtered) < len(words) / 3:
        return " ".join("".join(c for c in w if c.isalnum()) for w in words if w)
    return " ".join(filtered)

def smart_fuzzy_match(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    clean1 = remove_connective_fillers(text1)
    clean2 = remove_connective_fillers(text2)
    norm1 = normalize_for_comparison(clean1)
    norm2 = normalize_for_comparison(clean2)
    if norm1 == norm2:
        return 100.0
    scores = [
        fuzz.ratio(clean1, clean2),
        fuzz.token_sort_ratio(clean1, clean2),
        fuzz.token_set_ratio(clean1, clean2),
        fuzz.ratio(norm1, norm2),
        fuzz.token_sort_ratio(norm1, norm2),
        fuzz.token_set_ratio(norm1, norm2),
    ]
    w1, w2 = set(clean1.split()), set(clean2.split())
    if w1 and w2:
        inter = len(w1 & w2)
        uni = len(w1 | w2)
        if uni > 0:
            scores.append((inter / uni) * 100)
    word_scores = []
    for a in w1:
        for b in w2:
            if a == b:
                word_scores.append(len(a) * 10)
            elif len(a) > 3 and len(b) > 3:
                sim = fuzz.ratio(a, b)
                if sim > 80:
                    word_scores.append(sim * len(a) / 10)
    if word_scores:
        avg_word = sum(word_scores) / len(word_scores)
        scores.append(min(avg_word, 100))
    if not scores:
        return 0.0
    scores.sort(reverse=True)
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    weighted_sum = sum(sc * (weights[i] if i < len(weights) else 0.01) for i, sc in enumerate(scores))
    total_w = sum(weights[:len(scores)]) + 0.01 * max(0, len(scores) - len(weights))
    return min(weighted_sum / total_w, 100.0)

def get_matched_ids(
    pairs: List[Tuple[str, str]],
    reference_data: Dict[str, List[str]],
    threshold: int = 85,
) -> List[str]:
    """
    For each (section, act) in pairs, find the best matching ref id.
    Returns a list of ids with same length as pairs, 'NA' if no match >= threshold.
    """
    output_ids: List[str] = []
    for section_input, act_input in pairs:
        input_id, _ = find_section_identifier(section_input)
        best_score = 0.0
        best_id: Optional[str] = None

        for _, ref_value in reference_data.items():
            if len(ref_value) < 4:
                continue
            ref_section, ref_act, _, ref_id = ref_value[0], ref_value[1], ref_value[2], ref_value[3]
            ref_identifier, _ = find_section_identifier(ref_section)
            if not ref_identifier or not input_id:
                continue

            low_in, low_ref = input_id.lower(), ref_identifier.lower()
            matched_key = (
                low_in == low_ref
                or (len(low_in) > 1 and low_in[-1].isalpha() and low_in[:-1] == low_ref)
            )
            if not matched_key:
                continue

            clean_input_act = remove_connective_fillers(act_input.lower()) if act_input else ""
            clean_ref_act = remove_connective_fillers(ref_act.lower()) if ref_act else ""
            act_score = (
                smart_fuzzy_match(clean_input_act, clean_ref_act)
                if clean_input_act and clean_ref_act
                else (100.0 if not clean_input_act and not clean_ref_act else 50.0)
            )

            if act_score > best_score:
                best_score = act_score
                best_id = ref_id

        output_ids.append(best_id if best_score >= threshold and best_id else "NA")
    return output_ids

def process_model_output_to_ids(
    model_output,
    reference_json_path: str,
    threshold: int = 85,
) -> List[str]:
    with open(reference_json_path, "r", encoding="utf-8") as f:
        reference_data = json.load(f)
    pairs = extract_section_act_pairs_intelligent(model_output)
    return get_matched_ids(pairs, reference_data, threshold)

if __name__ == "__main__":
    input_json_path = r"/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/Rebuttal_ARR_Oct_2025/restricted/title-only/RAG_gpt_4-1_kStatfrom_title-only_QQ.json"
    reference_json_path = r"/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dataset/id-section-meta.json"
    threshold = 40 

    # mode = 0 -> use 'citations', output 'citations-id'
    # mode = 1 -> use 'raw_generation', output 'matched_ids', rename 'id' to 'query_no' if present
    mode = 1
    if len(sys.argv) > 1:
        mode = int(sys.argv[1])

    if mode == 0:
        key_name = "citations"
        output_key = "citations-id"
    else:
        key_name = "raw_generation"
        output_key = "matched_ids"

    with open(input_json_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    with open(reference_json_path, "r", encoding="utf-8") as fref:
        reference_data = json.load(fref)

    for obj in data:
        key_value = obj.get(key_name, [])

        if isinstance(key_value, str):
            key_value = parse_raw_generation_flexible(key_value)

        pairs = extract_section_act_pairs_intelligent(key_value)
        ids = get_matched_ids(pairs, reference_data, threshold)
        obj[output_key] = ids

        if key_name == "raw_generation":
            obj["raw_generation_parsed"] = key_value

        if mode == 1 and "id" in obj:
            obj["query_no"] = obj.pop("id")

    output_dir = os.path.dirname(input_json_path)
    inbase = os.path.splitext(os.path.basename(input_json_path))[0]
    outname = os.path.join(output_dir, f"id-{inbase}.json")
    with open(outname, "w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)

    print(f"Processed! Output written to: {outname}")
