import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os, json
import re
from tqdm import tqdm
import time

from two_shot_examples import choice_examples, single_step_examples, multi_step_examples

# Configuration
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
data_dir = "data"
image_root = 'your/image/root'
cache_dir = 'your/cache/dir'
model_path = "your/model/path"
results_dir = "your/result/dir/internvl.json"

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    ), key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = [
        resized_img.crop((
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )) for i in range(blocks)
    ]
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map.update({
        'vision_model': 0,
        'mlp1': 0,
        'language_model.model.tok_embeddings': 0,
        'language_model.model.embed_tokens': 0,
        'language_model.output': 0,
        'language_model.model.norm': 0,
        'language_model.model.rotary_emb': 0,
        'language_model.lm_head': 0,
        f'language_model.model.layers.{num_layers - 1}': 0
    })
    return device_map

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

# Load model and tokenizer
name = 'OpenGVLab/InternVL3-8B'
device_map = split_model('InternVL3-8B')
model = AutoModel.from_pretrained(
    name,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map,
    cache_dir=cache_dir
).eval()
tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=False, cache_dir=cache_dir)

def infer_sample(sample, prompt_type="default"):
    qa_id = sample['qa_id']
    question = generate_prompt_content(sample, prompt_type)
    question = re.sub(r'<image\d*>|<ImageHere>', '<image>', question)
    image_count = question.count('<image>')
    if image_count == 0 and len(sample['image']) > 0:
        question += ''.join(['<image>' for _ in sample['image']])
    elif image_count > len(sample['image']):
        question = '<image>'.join(question.split('<image>')[:len(sample['image'])+1])
    elif image_count < len(sample['image']):
        raise ValueError(f"Mismatch between <image> tags and image count in qa_id {qa_id}")

    question += " please answer the question step by step, and give the final answer at the end."
    pixel_values_list, num_patches_list = [], []
    for img_path in sample['image']:
        abs_path = os.path.join(image_root, img_path)
        img_tensor = load_image(abs_path).to(torch.bfloat16).cuda()
        pixel_values_list.append(img_tensor)
        num_patches_list.append(img_tensor.size(0))

    pixel_values = torch.cat(pixel_values_list, dim=0)
    start_time = time.time()
    response = model.chat(tokenizer, pixel_values, question, dict(max_new_tokens=2048, do_sample=False),
                          num_patches_list=num_patches_list, history=None, return_history=False)
    time_elapsed = time.time() - start_time
    generated_tokens = tokenizer(response, return_tensors='pt')['input_ids'].size(1)
    tokens_per_second = generated_tokens / time_elapsed if time_elapsed > 0 else float('inf')

    print(f'qa_id: {qa_id}\nQ: {question}\nA: {response}\n')
    print(f"Generated {generated_tokens} tokens in {time_elapsed:.2f}s ({tokens_per_second:.2f} tokens/sec)")

    return {
        "qa_id": qa_id,
        "question": sample['question'],
        "result": response,
        "answer": sample['answer'],
        "source": sample['source'],
        "answer_type": sample['answer_type'],
        "problem_type": sample['problem_type'],
        "complexity_level": sample['complexity_level'],
        "tokens_generated": generated_tokens,
        "time_elapsed": time_elapsed,
        "tokens_per_second": tokens_per_second
    }

# Main
input_file = os.path.join(data_dir, 'solid_geo_final.json')
prompt_type = "default"  # "default", "cot", "two-shot"
output_file = os.path.join(results_dir, f'InternVL-{prompt_type}.json')

with open(input_file, 'r') as f:
    data = json.load(f)

results = []
for key, sample in tqdm(data.items(), desc="Processing Samples"):
    try:
        results.append(infer_sample(sample, prompt_type=prompt_type))
    except Exception as e:
        print(f"Error processing qa_id {sample['qa_id']}: {e}")

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Results saved to {output_file}")
