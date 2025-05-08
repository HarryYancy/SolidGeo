# First run the following command to start the server:
# vllm serve /path/to/model --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --limit_mm_per_prompt 'image=10' --tensor-parallel-size 2
# Then run the following command in the image root directory to start the file server:
# python3 -m http.server 8001

import os
import argparse
import json
from tqdm import tqdm
import pandas as pd
import requests

# Import 2-shot examples
from two_shot_examples import choice_examples, single_step_examples, multi_step_examples


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


def main():
    # Argument parsing
    # Example command python3 Mistral-Small-3.1-24B-Instruct-2503.py default 2 3 6 7
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt_type", type=str, default="default", help="prompt type")
    parser.add_argument("dev", nargs='+', type=int, help="device")
    args = parser.parse_args()

    PROMPT_TYPE = args.prompt_type
    dev = ""
    for ele in args.dev:
        dev += str(ele) + ","
    dev = dev[:-1]
    # print(dev)
    os.environ['CUDA_VISIBLE_DEVICES'] = dev
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Model configuration
    MODEL_NAME = "Mistral-Small-3.1-24B-Instruct-2503"
    # Model configuration
    model_dir = f"/path/to/model"

    # Path configuration
    # Path configuration
    input_json = "/path/to/json_file.json"
    output_file = f"/path/to/{MODEL_NAME}_results_{PROMPT_TYPE}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    all_results = []

    data = pd.read_json(input_json, orient='index')

    # Read intermediate results from the output file
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        print(f"Intermediate results read from output file, total {len(all_results)} entries")

    # Extract all processed qa_ids
    processed_qa_ids = {result['qa_id'] for result in all_results}

    # Filter out processed rows
    data = data[~data['qa_id'].isin(processed_qa_ids)]

    pbar = tqdm(total=len(data), desc="Processing", unit="sample")
    cnt = 0
    tokens_cnt = 0
    for _, sample in data.iterrows():
        try:
            sample = sample.to_dict()
            qa_id = sample["qa_id"]
            question = generate_prompt_content(sample, PROMPT_TYPE)
            answer = sample.get("answer", "")
            source = sample.get("source", "")
            image_paths = [img.replace("\\", "/") for img in sample.get("image", [])]
            answer_type = sample.get("answer_type", "")
            problem_type = sample.get("problem_type", [])
            complexity_level = sample.get("complexity_level", "")

            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image_url", "image_url": {"url": f"http://localhost:8001/{path}"}} for path in image_paths],
                        {"type": "text", "text": question}
                    ]
                }]
            # print(messages)

            url = "http://127.0.0.1:8000/v1/chat/completions"
            headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}

            data = {"model": model_dir, "messages": messages, "temperature": 0.2}

            response = requests.post(url, headers=headers, data=json.dumps(data)).json()

            output_text = response["choices"][0]["message"]["content"]
            token_count = response['usage']['completion_tokens']
            # Record result
            result = {
                "qa_id": qa_id,
                "question": question,
                "result": output_text,
                "answer": answer,
                "source": source,
                "answer_type": answer_type,
                "problem_type": problem_type,
                "complexity_level": complexity_level,
            }
            all_results.append(result)

            # Count total tokens
            tokens_cnt += token_count

            print(f"[QA_ID: {qa_id}]\nQ: {question}\nA: {output_text}\nGT: {answer}\n")
            pbar.set_postfix({"QA_ID": qa_id, "Status": "Success", "avg_Tokens": tokens_cnt / (pbar.n + 1)})
            pbar.update(1)
            cnt += 1
            if cnt % 10 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                print(f"\nIntermediate results saved to: {output_file}")

            # del outputs
            # torch.cuda.empty_cache()

        except Exception as e:
            error_result = f"[ERROR] QA_ID: {qa_id} - {str(e)}"
            print(error_result)
            pbar.set_postfix({"Status": "Error"})
            pbar.update(1)
            continue

    pbar.close()
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nAll results saved to: {output_file}")


if __name__ == "__main__":
    main()
