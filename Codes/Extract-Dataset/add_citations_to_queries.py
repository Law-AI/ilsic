# postprocess_build_citations.py

import json
import re
from collections import defaultdict



LLM_RESPONSE_JSON = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dev/extract_dataset/output_test/extracted_sections_acts_cases.json"
QUERY_LEVEL_JSON = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dev/extract_dataset/combine.json"
OUTPUT_JSON = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dev/extract_dataset/output_test/queries_with_citations.json"


ACT_ABBREVIATIONS = {
    'constitution': 'constitution of india',
    'pwdv': 'protection of women from domestic violence act',
    'dva': 'protection of women from domestic violence act',
    'dv ': 'protection of women from domestic violence act',
    'mrtp': 'monopolies and restrictive trade practices act',
    'nia': 'negotiable instruments act',
    'ni ': 'negotiable instruments act',
    'mtal': 'maharastra tenancy and agricultural lands act',
    'hma': 'hindu marriage act',
    'rcr': 'hindu marriage act',
    'hsa': 'hindu succession act',
    'hm ': 'hindu marriage act',
    'h m': 'hindu marriage act',
    'sma': 'special marriage act',
    'sarfaesi': 'securitisation and reconstruction of financial assets and enforcement of security interest act',
    'sarfesi': 'securitisation and reconstruction of financial assets and enforcement of security interest act',
    'cpc': 'code of civil procedure',
    'crpc': 'code of criminal procedure',
    'rera': 'real estate (regulation and development) act',
    'rti': 'right to information act',
    'it ': 'income tax act',
    'dpa': 'dowry prohibition act',
    'dp ': 'dowry prohibition act',
    'mva': 'motor vehicles act',
    'mv ': 'motor vehicles act',
    'mofa': 'maharashtra ownership flats act',
    'bmc': 'bombay municipal corporation act',
    'tpa': 'transfer of property act',
    'tp ': 'transfer of property act',
    'mc': 'municipal corporation act',
    'cgst': 'central goods and services tax act',
    'gst': 'central goods and services tax act',
    'scst': 'scheduled castes and scheduled tribes (prevention of atrocities) act',
    'sc st': 'scheduled castes and scheduled tribes (prevention of atrocities) act',
    'mcs': 'maharashtra cooperative societies act',
    'mmc': 'maharashtra municipal corporations act',
    'ida': 'industrial disputes act',
    'id ': 'industrial disputes act',
    'posco': 'protection of children from sexual offences act',
    'fea': 'foreign exchange management act',
    'llp': 'limited liability partnership act',
    'ulc': 'urban land (ceiling and regulation) act',
    'fssai': 'food safety and standards act',
    'rbi': 'reserve bank of india act',
    'upzalr': 'uttar pradesh zamindari abolition and land reforms act',
    'ipc': 'indian penal code',
    'pc ': 'prevention of corruption act',
    'ibc': 'insolvency and bankruptcy code',
    'kofa': 'karnataka ownership flats act',
    'kaoa': 'karnataka apartment ownership act',
    'ksra': 'karnataka stamp act',
    'esi': 'employees state insurance act',
    'kuda': 'karnataka urban development authority act',
    'klr': 'kerala land reforms act',
    'dmc': 'delhi municipal corporation act',
    'pmla': 'prevention of money laundering act',
    'cotpa': 'cigarettes and other tobacco products act',
    'mrca': 'maharashtra rent control act',
    'rddb': 'recovery of debts due to banks and financial institutions act',
    'sebi': 'securities and exchange board of india act',
    'hama': 'hindu adoption and maintenance act',
    'ham': 'hindu adoption and maintenance act',
    'fema': 'foreign exchange management act',
    'fcr': 'foreign contribution regulation act',
    'la': 'limitation act',
}


CANONICAL_ACT_PATTERNS = [
    (re.compile(r"\b(d\.?\s*v\.?\s*c|d\.?\s*v\.?\s*act)\b", re.I),
     "protection of women from domestic violence act"),
    (re.compile(r"\b(domestic\s+violence\s+act)(\s*,?\s*2005)?\b", re.I),
     "protection of women from domestic violence act"),
    (re.compile(r"\b(ipc|indian\s+penal\s+code)\b", re.I),
     "indian penal code"),
    (re.compile(r"\b(crpc|cr\.?\s*p\.?\s*c\.?|code\s+of\s+criminal\s+procedure)\b", re.I),
     "code of criminal procedure"),
    (re.compile(r"\b(cpc|code\s+of\s+civil\s+procedure)\b", re.I),
     "code of civil procedure"),
]



IPC_ONLY_SECTIONS = {
    "498", "498a", "420", "406", "405", "409", "415", "506"
}

DV_ALLOWED_SECTIONS = {
    "12", "17", "18", "19", "20", "21", "22", "23", "31", "33"
}



def normalize_act(act: str) -> str:
    if not act:
        return None

    text = act.lower().strip()
    text = re.sub(r"[^\w\s\.]", "", text)

    for pattern, canon in CANONICAL_ACT_PATTERNS:
        if pattern.search(text):
            return canon

    return text


def normalize_section(section: str) -> str:
    if not section:
        return None

    s = section.lower().strip()
    s = re.sub(r"^(section|sec|sections)\s*", "", s)
    s = re.sub(r"[^\w\(\)\-to]", "", s)

    if not re.search(r"\d", s):
        return None

    return s.strip()


def is_valid_pair(section: str, act: str) -> bool:
    """
    Enforce section–act compatibility
    """
    if act == "protection of women from domestic violence act":
        return section in DV_ALLOWED_SECTIONS

    if act == "indian penal code":
        return section in IPC_ONLY_SECTIONS or section.isdigit()

    return True

def expand_section_range(section: str):
    """
    Expands:
      10-12     → [10, 11, 12]
      10 to 12  → [10, 11, 12]
      156(3)    → [156(3)]
    """
    if not section:
        return []

    s = section.strip()

    # Match ranges like "12-14" or "12 to 14"
    m = re.match(r"^(\d+)\s*(?:-|to)\s*(\d+)$", s)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if start <= end and (end - start) <= 50:  # safety cap
            return [str(i) for i in range(start, end + 1)]

    # Otherwise return as-is
    return [s]


def make_citation(section: str, act: str) -> str:
    return f"{section} {act}"



with open(LLM_RESPONSE_JSON, "r", encoding="utf-8") as f:
    responses = json.load(f)

with open(QUERY_LEVEL_JSON, "r", encoding="utf-8") as f:
    queries = json.load(f)


query_sections_raw = defaultdict(list)

for row in responses:
    qid = row.get("query_id")

    for sec in row.get("sections", []):
        act = normalize_act(sec.get("act"))
        raw_section = normalize_section(sec.get("section"))

        if not act or not raw_section:
            continue

        for section in expand_section_range(raw_section):
            if not is_valid_pair(section, act):
                continue

            query_sections_raw[qid].append((section, act))




query_citations = {}

for qid, pairs in query_sections_raw.items():
    unique = {make_citation(sec, act) for sec, act in pairs}
    query_citations[qid] = sorted(unique)



for q in queries:
    qid = q.get("id") or q.get("query_id")
    q["citations"] = query_citations.get(qid, [])



with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(queries, f, ensure_ascii=False, indent=2)

print(f"[INFO] Canonical citations added and saved to {OUTPUT_JSON}")
