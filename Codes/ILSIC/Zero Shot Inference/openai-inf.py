import json
import os
import lzma
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


OPENAI_API_KEY = "OpenAI API key"  # put your key
MODEL = "gpt-4.1-mini"
client = OpenAI(api_key=OPENAI_API_KEY)

TEST_INPUT_PATH = "/home/shounak/HDD/Layman-LSI/dataset/20Q.json"
OUTPUT_PATH = "/home/shounak/HDD/Layman-LSI/QA/base-inf-qa/gpt4-1-mini_20q_qa.json"

INPUT_FORMAT  = "auto"
OUTPUT_FORMAT = "auto"

ID_FIELD     = "id"
INPUT_FIELD  = "query-text"  # <- your example uses this

SYSTEM_PROMPT = """ 
You are a legal expert on Indian law. Given the text below, analyze it and provide the answer strictly in this structure:
[ 
"Statute Reference: <fill here>"; 
"Legal Issues: <fill here>";  
"Procedural Advice: <fill here>"; 
]
Instructions:
1. Always output only this one list — nothing before or after.
2. Each field must be a quoted string starting with the field name exactly as written.
3. Separate fields with semicolons.
4. Fill every field concisely with the most relevant information. If no clear answer exists, write "None".
5. Use only provisions and concepts clearly supported by Indian law. No speculation, no abbreviations, no duplicates.
"""

def infer_format_from_path(path: str) -> str:
    p = path.lower()
    if ".jsonl" in p:  # matches .jsonl and .jsonl.xz
        return "jsonl"
    return "json"      # default

def compression_for(fmt: str):
    """Your rule: JSONL => xz, JSON => None."""
    return "xz" if fmt.lower() == "jsonl" else None

def whitespace_handler(text):
    import re
    text = re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', text.strip()))
    text = text.replace("\xad", "")
    return text

def openai_chat_completion_response(case_text):
    user_prompt = "Facts of the case: " + whitespace_handler(case_text)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip(" \n")

def read_input_dataframe(path: str, fmt: str) -> pd.DataFrame:
    fmt = fmt.lower()
    comp = compression_for(fmt)
    lines = (fmt == "jsonl")
    return pd.read_json(path, lines=lines, compression=comp)

def save_results(results, path: str, fmt: str):
    fmt = fmt.lower()
    comp = compression_for(fmt)
    if fmt == "json":
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    elif fmt == "jsonl":
        
        if comp == "xz":
            with lzma.open(path, "wt", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            with open(path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        raise ValueError("OUTPUT_FORMAT must be 'json' or 'jsonl'.")
    print(f"\n==> Saved {len(results)} results to {path}\n")

def batch_inference_and_save(records, output_path, fmt, batch_size=10):
    results = []
    last_saved = 0
    for idx, rec in enumerate(tqdm(records, total=len(records), desc="OpenAI Inference", ncols=100), start=1):
        input_text = rec.get(INPUT_FIELD)
        if input_text is None:
            # common fallbacks
            for k in ("instruction", "input", "text", "prompt"):
                if k in rec:
                    input_text = rec[k]; break
        if input_text is None:
            print(f"Warning: missing input field at idx {idx}; skipping.")
            continue

        try:
            output_text = openai_chat_completion_response(input_text)
        except Exception as e:
            print(f"Error at idx {idx}: {e}")
            output_text = ""

        results.append({
            "query_no": rec.get(ID_FIELD, idx),
            "input": input_text,
            "model_output": output_text
        })

        if idx % batch_size == 0:
            save_results(results, output_path, fmt)
            last_saved = idx

    if last_saved < len(records):
        save_results(results, output_path, fmt)


if __name__ == "__main__":
    
    in_fmt  = infer_format_from_path(TEST_INPUT_PATH)  if INPUT_FORMAT  == "auto" else INPUT_FORMAT.lower()
    out_fmt = infer_format_from_path(OUTPUT_PATH)      if OUTPUT_FORMAT == "auto" else OUTPUT_FORMAT.lower()

    df = read_input_dataframe(TEST_INPUT_PATH, in_fmt)
    records = df.to_dict(orient="records")

    batch_inference_and_save(records, OUTPUT_PATH, out_fmt, batch_size=10)
