import os
import torch
import re
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MODEL_ID = "google/gemma-3-12b-it"
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

TEST_JSONL_XZ= "/home/shounak/HDD/Layman-LSI/dataset/Layman-new-dataset/FT-Layman-test.jsonl.xz"
OUTPUT_DIR = "/home/shounak/HDD/Layman-LSI/model/Gemma_3_C&Ls_FT_bt_1_lr_1e5/"
PEFT_ID = "checkpoint-2898"
PEFT_PATH = os.path.join(OUTPUT_DIR, PEFT_ID)
OUTPUT_JSON_PATH = "/home/shounak/HDD/Layman-LSI/inf_result/gemma3/Court+LMsub/C+LMsub_test_new_inf_2898.json"
LONG_CONTEXT_WINDOW = 32000  

print(MODEL_ID)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.model_max_length = LONG_CONTEXT_WINDOW

bits_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bits_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir="/home/shared/hf_cache"
)
model = PeftModel.from_pretrained(base_model, PEFT_PATH, device_map="auto")
model.eval()
model.config.use_cache = True

def whitespace_handler(text):
    text = re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', text.strip()))
    text = text.replace("\xad", "")
    return text

def build_prompt_llama3(tokenizer, source, target=None):
    system_message = (
        """
        You are a legal expert on Indian law. Given the text below, identify all applicable legal sections and their full act names. Follow these rules exactly:
        1. Identify every relevant provision (section number + full act name) you are 100% certain applies under Indian statutes.
        2. Output only one list in square brackets, formatted like: ["Section X of Act Name"; "Section Y of Act Name"; …]
            - Each entry must be a quoted string with the section first, then the full act name.
            - Each entry needs both section number AND full act name
            - Separate entries with a semicolon.
            - Do not include anything outside this single list.
        3. Exclude any entry missing either section or act (no incomplete pairs).
        4. Do not repeat identical section-act pairs (no duplicates).
        5. Do not add explanations, labels, or extra text—only the list itself.
        6. Always spell out the full name of each act (no abbreviations).
        7. Include only provisions that are clearly and directly relevant (no speculative or uncertain entries).
        """
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": whitespace_handler(source)},
    ]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return chat_prompt.strip()

def build_prompt_gemma3(tokenizer, source, target=None):
    system_message = (
        """
        You are a legal expert on Indian law. Given the text below, identify all applicable legal sections and their full act names. Follow these rules exactly:
        1. Identify every relevant provision (section number + full act name) you are 100% certain applies under Indian statutes.
        2. Output only one list in square brackets, formatted like: ["Section X of Act Name"; "Section Y of Act Name"; …]
            - Each entry must be a quoted string with the section first, then the full act name.
            - Each entry needs both section number AND full act name
            - Separate entries with a semicolon.
            - Do not include anything outside this single list.
        3. Exclude any entry missing either section or act (no incomplete pairs).
        4. Do not repeat identical section-act pairs (no duplicates).
        5. Do not add explanations, labels, or extra text—only the list itself.
        6. Always spell out the full name of each act (no abbreviations).
        7. Include only provisions that are clearly and directly relevant (no speculative or uncertain entries).
        """
    )
    user_message = f"{system_message}\n\n{whitespace_handler(source)}"
    messages = [
        {"role": "user", "content": user_message},
    ]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return chat_prompt.strip()

def get_prompt_builder(model_id):
    if "llama" in model_id.lower():
        return build_prompt_llama3
    elif "gemma" in model_id.lower():
        return build_prompt_gemma3
    else:
        raise ValueError("Unknown model type (only supports Llama-3 and Gemma-3)")

prompt_builder = get_prompt_builder(MODEL_ID)

def infer(model, tokenizer, instruction, temperature=0.1, max_new_tokens=1024, num_beams=4):
    prompt = prompt_builder(tokenizer, instruction)
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length
    ).to(device)
    input_length = inputs['input_ids'].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            min_new_tokens=32,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=num_beams,
            no_repeat_ngram_size=8,
            early_stopping=True
        )
    generated = outputs[0][input_length:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()

def batch_inference_and_save(test_df, output_json_path, batch_size=10):
    results = []
    last_saved_id = 0

    def save_results(results_to_save, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)
        print(f"\n==> Saved {len(results_to_save)} results to {path}\n")

    for idx, row in enumerate(tqdm(test_df.itertuples(), total=len(test_df), desc="Inferencing", ncols=100)):
        input_text = row.instruction
        output_text = infer(model, tokenizer, input_text)
        results.append({
            "query_no": idx + 1,
            "input": input_text,
            "model_output": output_text
        })

        if (idx + 1) % batch_size == 0:
            save_results(results, output_json_path)
            last_saved_id = idx + 1

    if last_saved_id < len(test_df):
        save_results(results, output_json_path)

if __name__ == "__main__":
    test_df = pd.read_json(
        TEST_JSONL_XZ,
        lines=True,
        compression="xz"
    )
    batch_inference_and_save(test_df, OUTPUT_JSON_PATH, batch_size=10)
