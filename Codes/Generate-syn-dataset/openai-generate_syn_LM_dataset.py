import json
import os
from tqdm import tqdm
from openai import OpenAI

# ========== CONFIG ==========
OPENAI_API_KEY = "OpenAI key" 
MODEL = "gpt-4.1"      
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are a legal expert specializing in Indian court cases. I will provide you with the facts of a court case. Your task is to generate a layman's summary of the case, written in a single, continuous paragraph, without using bullet points, numbered lists, or section headers. The summary should be clear, concise, and easy to understand for someone without legal expertise, but it must include all relevant facts and details. Write in a conversational style, as if you are personally describing your situation and seeking advice, similar to how a layperson might post their question on an online forum.

Important: Do not label or mention legal entities, sections, acts, or precedents with placeholders such as [ENTITY], [SECTION], [ACT], [PRECEDENT], etc. Do not add, invent, or fill in any legal names, sections, or references that are not present in the original case facts I provide. If the court facts I provide do not mention a statute (such as a section or act), you must not include any such statute in your output, under any circumstance. Only summarize the information actually present in the input.

Only output the summary in the required style.
"""

def openai_chat_completion_response(case_text):
    user_prompt = "Facts of the case: " + case_text.strip()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
    )
    
    return response.choices[0].message.content.strip(" \n")

def convert_dataset(input_path, output_path):
    with open(input_path) as fr:
        data = json.load(fr)
    print(f"\n=== Processing dataset: {input_path} ===")
    print(f"Total queries to process: {len(data)}")
    with tqdm(total=len(data), unit="query", ncols=80, desc="Progress") as pbar:
        for idx, entry in enumerate(data):
            case_text = entry["query-text"]
            try:
                summary = openai_chat_completion_response(case_text)
            except Exception as e:
                print(f"Error at idx {idx} (id={entry['id']}): {e}")
                summary = ""  
            entry["query-text"] = summary
            
            if (idx + 1) % 10 == 0 or (idx + 1) == len(data):
                with open(output_path, "w") as fw:
                    json.dump(data, fw, indent=2, ensure_ascii=False)
                print(f"\n{idx + 1} queries processed and saved in {output_path}")
            pbar.update(1)
    print(f"=== Finished processing {input_path} ===\n")
    print(f"Saved output to {output_path}")


dataset_files = [
    ("id-court-dev-final-clean.json", "id-syn-layman-dev.json"),
    ("id-court-test-final-clean.json", "id-syn-layman-test.json"),
    ("id-court-train-final-clean.json", "id-syn-layman-train.json"),
]

base_in_path = "/home/shounak/HDD/Layman-LSI/dataset/court-dataset/"
base_out_path = "/home/shounak/HDD/Layman-LSI/dataset/synthetic-layman-dataset/"

for in_file, out_file in dataset_files:
    input_path = os.path.join(base_in_path, in_file)
    output_path = os.path.join(base_out_path, out_file)
    convert_dataset(input_path, output_path)
