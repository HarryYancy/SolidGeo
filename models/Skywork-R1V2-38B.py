import os
import argparse
import json
from tqdm import tqdm
import pandas as pd
from PIL import Image
from typing import List, Union
# Import 2-shot examples
from two_shot_examples import choice_examples, single_step_examples, multi_step_examples


def load_images(image_paths: List[str]) -> Union[Image.Image, List[Image.Image]]:
    images = [Image.open(img_path) for img_path in image_paths]
    return images[0] if len(images) == 1 else images

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

def prepare_question(question: str, num_images: int) -> str:
    if not question.startswith("<image>\n"):
        return "<image>\n" * num_images + question
    return question


def main():
    # Argument parsing
    # Example command python3 Skywork-R1V2-38B.py default 2 3 6 7
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt_type", type=str, default="default", help="prompt type")
    parser.add_argument("dev", nargs='+', type=int, help="device")
    args = parser.parse_args()

    PROMPT_TYPE=args.prompt_type
    dev=""
    for ele in args.dev:
        dev+=str(ele)+","
    dev=dev[:-1]

    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = dev
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Model configuration
    MODEL_NAME = "Skywork-R1V2-38B"
    # Model configuration
    model_dir = f"/path/to/model"

    # Path configuration
    input_json = "/path/to/json_file.json"
    image_root = "/path/to/image_root/"
    output_file = f"/path/to/{MODEL_NAME}_results_{PROMPT_TYPE}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    

    from vllm import LLM,SamplingParams

    from transformers import AutoProcessor, LogitsProcessor
    import torch
    
    model = LLM(
            model=model_dir,
            tensor_parallel_size=len(args.dev),
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 10},
            gpu_memory_utilization=0.9,
        )
    
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=16384,
        repetition_penalty=1.2,
    )

    processor = AutoProcessor.from_pretrained(model_dir,
                                          use_fast=True,
                                          trust_remote_code=True,
                                          local_files_only=True,
                                          min_pixels=256 * 28 * 28,
                                          max_pixels=1280 * 28 * 28)



    # Inference settings
    all_results = []

    data=pd.read_json(input_json,orient='index')


    # Read intermediate results from output file
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        print(f"Read intermediate results from output file, total {len(all_results)} samples")
        
    # Extract all processed qa_ids
    processed_qa_ids = {result['qa_id'] for result in all_results}

    # Filter out rows that have already been processed
    data = data[~data['qa_id'].isin(processed_qa_ids)]


    pbar = tqdm(total=len(data), desc="Processing", unit="sample")
    cnt=0
    tokens_cnt=0
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


            question = prepare_question(question, len(image_paths))
            messages = [{
                "role": "user",
                "content": question
            }]
            # print(messages)
            # Build input
            

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs= load_images(image_paths)


            outputs = model.generate(
            {
            "prompt": text,
            "multi_modal_data": {"image": image_inputs},
            
            },
            sampling_params=sampling_params
            )

            
            output_text = outputs[0].outputs[0].text.strip()
            token_count = len(outputs[0].outputs[0].token_ids)
            # Record results
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
            tokens_cnt+=token_count

            pbar.set_postfix({"QA_ID":qa_id,"Status": "Success", "avg_Tokens":tokens_cnt/(pbar.n+1)})
            pbar.update(1)
            cnt+=1
            if cnt%10==0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                print(f"\nIntermediate results saved to: {output_file}")

            del outputs
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

if __name__ == "__main__":
    
    main()
