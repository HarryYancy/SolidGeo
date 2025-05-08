
import os
import argparse


# Parameter parsing
# Example command python3 {file} default 2 3 6 7
parser = argparse.ArgumentParser()
parser.add_argument("prompt_type", type=str, default="default", help="prompt type")
parser.add_argument("dev", nargs='+', type=int, help="device")
args = parser.parse_args()

PROMPT_TYPE=args.prompt_type
MODEL_NAME="R1-Onevision-7B-RL"
dev=""
for ele in args.dev:
    dev+=str(ele)+","
dev=dev[:-1]

os.environ['CUDA_VISIBLE_DEVICES'] = dev

# Model configuration
model_dir = f"/path/to/model"

# Path configuration
input_json = "/path/to/json_file.json"
image_root = "/path/to/image_root/"
output_file = f"/path/to/{MODEL_NAME}_results_{PROMPT_TYPE}.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

from transformers import  AutoProcessor, LogitsProcessor, AutoModelForVision2Seq, AutoModel, AutoTokenizer
from qwen_vl_utils import process_vision_info

import torch
import json
from tqdm import tqdm
import pandas as pd

# Introduce 2-shot examples
from two_shot_examples import choice_examples, single_step_examples, multi_step_examples


class NanInfCheckProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
            print("Detected invalid logits, safely processed.")
        return scores

def generate_prompt_content(sample, prompt_type="default"):
    q = sample["question_input"]
    t = sample.get("answer_type", "default")
    if prompt_type == "default":
        return q
    elif prompt_type == "cot":
        return q.strip() + "\nPlease reason step by step before answering."
    elif prompt_type == "two-shot":
        prefix = {
            "choice": choice_examples,
            "single-step": single_step_examples,
            "multi-step": multi_step_examples
        }.get(t, "")
        return prefix + "\n\nQuestion: " + q
    return q


model = AutoModelForVision2Seq.from_pretrained(
    model_dir,
    # torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    local_files_only=True
)

processor = AutoProcessor.from_pretrained(model_dir,
                                          use_fast=True,
                                          trust_remote_code=True,
                                          local_files_only=True,
                                          min_pixels=256 * 28 * 28,
                                          max_pixels=1280 * 28 * 28)


# Inference settings
logits_processor = [NanInfCheckProcessor()]


all_results = []


data=pd.read_json(input_json,orient='index')


# Read intermediate results from output file
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    print(f"Intermediate results read from output file, total {len(all_results)} entries")
    
# Extract all processed qa_id
processed_qa_ids = {result['qa_id'] for result in all_results}

# Filter out processed rows
data = data[~data['qa_id'].isin(processed_qa_ids)]


pbar = tqdm(total=len(data), desc="Processing", unit="sample")
cnt=0
for _, sample in data.iterrows():
    try:
        sample = sample.to_dict()
        qa_id = sample["qa_id"]
        question = generate_prompt_content(sample, PROMPT_TYPE)
        answer = sample.get("answer", "")
        source = sample.get("source", "")
        image_paths = [os.path.join(image_root, img.replace("\\", "/")) for img in sample.get("image", [])]
        answer_type = sample.get("answer_type","")
        problem_type = sample.get("problem_type",[])
        complexity_level = sample.get("complexity_level","")

        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": path} for path in image_paths],
                {"type": "text", "text": question}
            ]
        }]

        # Build input
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        # Model generation
        generated_ids = model.generate(
            **inputs,
            temperature=0.2,
            max_new_tokens=5120,
            logits_processor=logits_processor,
            do_sample=True
        )

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()

        # Record result
        result = {
            "qa_id": qa_id,
            "question": question,
            "result": output_text,
            "answer": answer,
            "source": source,
            "answer_type": answer_type,
            "problem_type": problem_type,
            "complexity_level": complexity_level
        }
        all_results.append(result)

        print(f"[QA_ID: {qa_id}]\nQ: {question}\nA: {output_text}\nGT: {answer}\n")
        pbar.set_postfix({"QA_ID":qa_id,"Status": "Success", "Len": len(output_text)})
        pbar.update(1)

        cnt+=1
        if cnt%20==0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"\nIntermediate results saved to: {output_file}")

        del inputs, generated_ids
        torch.cuda.empty_cache()

    except Exception as e:
        error_result=f"[ERROR] QA_ID: {qa_id} - {str(e)}"
        print(error_result)
        pbar.set_postfix({"Status": "Error"})
        pbar.update(1)
        continue

pbar.close()
# Save results
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\nAll results saved to: {output_file}")
