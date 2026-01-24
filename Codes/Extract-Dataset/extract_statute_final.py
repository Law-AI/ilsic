import json
import time
import re
import pandas as pd
import tqdm
from openai import OpenAI

INPUT_JSON = "/home/shounak/Restored_Data/HDD1/shounak/Layman-LSI/dev/extract_dataset/combine.json"
FLAT_CSV = "legalqa.csv"


with open(INPUT_JSON, "r", encoding="utf-8") as file:
    data = json.load(file)

flattened_data = []

for query in data:
    query_base = {
        "query_id": query.get("id"),
        "query-title": query.get("query-title", ""),
        "query-text": query.get("query-text", ""),
        "query-url": query.get("query-url", ""),
        "query-category": query.get("query-category", ""),
        "query-religion": query.get("query-religion", ""),
    }

    for response in query.get("responses", []):
        flattened_data.append({
            **query_base,
            "responder": response.get("responder", ""),
            "response-text": response.get("response-text", ""),
        })

df = pd.DataFrame(flattened_data)
df.to_csv(FLAT_CSV, index=False)
print(f"[INFO] Flattened dataset saved: {FLAT_CSV}")

# -------------------- LLM PROMPT --------------------
LEGAL_EXTRACTION_PROMPT = """
You are a highly knowledgeable legal expert. Your task is to identify and extract all legal references, such as statutes_2993, acts, sections, and legal cases, explicitly mentioned in the input text. Follow these detailed instructions carefully:
    
    1. **Extract Only Mentioned References**: Identify and extract only the legal statutes_2993, acts, sections, and legal cases that are explicitly mentioned in the input text. Avoid inferring or including any information that is not directly stated in the text.
    
    2. **Categorize Extracted Information**: Organize the extracted references into three distinct categories:
       - **Acts**: Extract and list all acts mentioned in the text. Ensure the names follow the standard format used in official legal documentation. Replace "Indian Penal Code" with "IPC" and "Criminal Penal Code" with "CrPC" where applicable.
       - **Sections**: Extract and list all specific sections mentioned, along with their associated legal codes. Use abbreviations (e.g., "IPC" instead of "Indian Penal Code" and "CrPC" instead of "Criminal Penal Code") where applicable.
       - **Cases**: Extract and list all court cases explicitly mentioned in the text, including their full citation or any identifying details if provided.

    3. **Accuracy and Completeness**: Ensure your extraction is complete, accurate, and in adherence to the exact language of the input text. Avoid omitting or adding any information that is not explicitly mentioned.
    
    4. **Output in JSON Format**: Present the extracted information in JSON format with the following fields:
           {
            "sections": [
                {
                "act": "IPC or CrPC or other act",
                "section": "section number as mentioned in text"
                }
            ],
            "cases": [
                "list of cases"
            ]
            }

            Rules:
            - Each section MUST explicitly include its associated act
            - Do NOT output acts separately
            - If multiple sections belong to the same act, list them as separate objects
            - Do NOT infer acts; only use what is explicitly stated
           
    Each category should contain a list of extracted references. If no references are found for a category, use an empty list ([]) for that category.
    
    5. Maintain Legal Standards: Ensure the extracted acts, sections, and cases are formatted to match their official representation as per legal documentation. Indian Penal code should be written at IPC and Criminal Penal code as CrPC
    
    6. No Unmentioned Information: Do not include any acts, sections, or cases that are not explicitly mentioned in the input text. Avoid inferring or assuming references based on related information.
    
    7. Handling Edge Cases: If the input text contains partial references or ambiguous mentions, include them as-is but clearly label them in the structured output.

    8. Standardisation: When listing acts and sections:

    Replace "Indian Penal Code" with "IPC".
    Replace "Criminal Penal Code" with "CrPC".
    Ensure all names and abbreviations follow their official representation.
    
    Your output must adhere to the JSON format specified above, with each key ("acts", "sections", "cases") having their corresponding values as lists of extracted references.
    
"""

client = OpenAI(
    api_key="OPENAI API KEY"
)

def extract_legal_references(text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": LEGAL_EXTRACTION_PROMPT},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("[ERROR] OpenAI failure:", e)
        return "{}"

# -------------------- RUN EXTRACTION --------------------
OUTPUT_CSV = (
    "/home/shounak/Restored_Data/HDD1/shounak/"
    "Layman-LSI/dev/extract_dataset/output_test/"
    "extracted_statutes_cases.csv"
)

records = []

print("[INFO] Extracting from ALL responses:", df.shape[0])

for i, (idx, row) in enumerate(
    tqdm.tqdm(df.iterrows(), total=len(df), desc="Extracting legal references"),
    start=1
):
    time.sleep(7)
    result = extract_legal_references(row["response-text"])
    records.append({
        "query_id": row["query_id"],
        "response_index": idx,
        "response_text": row["response-text"],
        "results": result
    })

df_llm = pd.DataFrame(records)
df_llm.to_csv(OUTPUT_CSV, index=False)
print("[INFO] ChatGPT extraction saved")

# -------------------- RANGE EXPANSION (ENHANCED) --------------------
def expand_section_ranges(sections):
    expanded = []

    for item in sections:
        act = item.get("act")
        raw_sec = str(item.get("section")).lower().strip()

        # remove words like "section", "sections", "of ipc"
        raw_sec = re.sub(r"(section|sections|of|ipc|crpc|cpc)", "", raw_sec)
        raw_sec = raw_sec.strip()

        # match ranges: 10-12, 10–12, 10 to 12
        match = re.search(r"(\d+)\s*(?:-|–|to)\s*(\d+)", raw_sec)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            for s in range(start, end + 1):
                expanded.append({"act": act, "section": str(s)})
        else:
            # extract single number if present
            single = re.search(r"\d+", raw_sec)
            if single:
                expanded.append({"act": act, "section": single.group()})

    return expanded

# -------------------- PARSE OUTPUT --------------------
def extract_info(text):
    try:
        text = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE)
        data = json.loads(text)

        sections = expand_section_ranges(data.get("sections", []))
        cases = data.get("cases", [])

        return pd.Series([sections, cases])
    except Exception as e:
        print("[PARSE ERROR]", e)
        print(text)
        return pd.Series([[], []])

FINAL_OUTPUT = (
    "/home/shounak/Restored_Data/HDD1/shounak/"
    "Layman-LSI/dev/extract_dataset/output_test/"
    "extracted_sections_acts_cases.csv"
)

df = pd.read_csv(OUTPUT_CSV)
df[["sections", "cases"]] = df["results"].apply(extract_info)
df.to_csv(FINAL_OUTPUT, index=False)

print("[INFO] Final structured CSV saved")

# -------------------- FINAL JSON --------------------
FINAL_JSON_OUTPUT = (
    "/home/shounak/Restored_Data/HDD1/shounak/"
    "Layman-LSI/dev/extract_dataset/output_test/"
    "extracted_sections_acts_cases.json"
)

with open(FINAL_JSON_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

print("[INFO] Final structured JSON saved")
