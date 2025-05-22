import os
import json
import torch
import argparse
from tqdm import tqdm
import pandas as pd
from transformers import AutoProcessor, AutoModelForVision2Seq, LogitsProcessor
from qwen_vl_utils import process_vision_info
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


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Multi-Image Inference")
    parser.add_argument("prompt_type", type=str, choices=["default", "cot", "two-shot"], help="Prompt type")
    parser.add_argument("dev", nargs='+', type=int, help="CUDA visible devices")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--input_json", type=str, required=True, help="Path to input .json or .jsonl")
    parser.add_argument("--image_root", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output results")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    args = parser.parse_args()

    # Setup environment
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, args.dev))
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load model and processor
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        local_files_only=True
    )

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28
    )

    # Load data
    if args.input_json.endswith(".jsonl"):
        data = pd.read_json(args.input_json, lines=True)
    else:
        data = pd.read_json(args.input_json, orient="index")

    # Load existing results if any
    all_results = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} intermediate results from previous run.")
    processed_ids = {r['qa_id'] for r in all_results}
    data = data[~data['qa_id'].isin(processed_ids)]

    # Inference loop
    logits_processor = [NanInfCheckProcessor()]
    pbar = tqdm(total=len(data), desc="Processing")
    for _, row in data.iterrows():
        try:
            sample = row.to_dict()
            qa_id = sample["qa_id"]
            question = generate_prompt_content(sample, args.prompt_type)
            image_paths = [os.path.join(args.image_root, img) for img in sample.get("image", [])]
            messages = [{
                "role": "user",
                "content": [
                    *[{"type": "image", "image": path} for path in image_paths],
                    {"type": "text", "text": question}
                ]
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to("cuda")

            generated_ids = model.generate(
                **inputs,
                temperature=args.temperature,
                max_new_tokens=16384,
                logits_processor=logits_processor,
                do_sample=True
            )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            tokens_generated = generated_ids_trimmed[0].shape[-1]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()

            result = {
                "qa_id": qa_id,
                "question": question,
                "result": output_text,
                "answer": sample.get("answer", ""),
                "source": sample.get("source", ""),
                "answer_type": sample.get("answer_type", ""),
                "problem_type": sample.get("problem_type", ""),
                "complexity_level": sample.get("complexity_level", ""),
                "tokens_generated": tokens_generated
            }
            all_results.append(result)

            print(f"[QA_ID: {qa_id}]\nQ: {question}\nA: {output_text}\n")
            pbar.update(1)

            if len(all_results) % 20 == 0:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                print(f"Checkpoint saved at {args.output_file}")

            del inputs, generated_ids
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ERROR] QA_ID {sample.get('qa_id')} - {str(e)}")
            pbar.update(1)
            continue

    pbar.close()
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nAll results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
