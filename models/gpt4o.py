import os
import json
import base64
import time
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# Claude API 配置
api_key = 'your_api_key'
url = 'your_url'

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# 输入输出路径
input_json = "path_to_input_json"
image_root = "path_to_image_root"
output_json = "path_to_output_json/gpt4o.json"

# 读取数据
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

def encode_image_base64(image_path):
    """将图像编码为 base64 格式的字符串"""
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        base64_str = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"

results = []

for _, item in tqdm(data.items(), desc="Processing"):
    qa_id = item["qa_id"]
    question = item["question_input"]
    ground_truth = item["answer"]
    image_paths = item["image"]
    source = item.get("source")
    answer_type = item.get("answer_type")
    problem_type = item.get("problem_type")
    complexity_level = item.get("complexity_level")

    # 构造图文 content
    content = []
    content.append({"type": "text", "text": question})
    for img_file in image_paths:
        full_path = os.path.join(image_root, img_file)
        if os.path.exists(full_path):
            image_b64 = encode_image_base64(full_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_b64
                }
            })
        else:
            content.append({
                "type": "text",
                "text": f"[Image missing: {img_file}]"
            })

    # 构造 payload
    payload = {
        "model": "gpt-4o-2024-05-13", 
        "messages": [
            {
                "role": "system",
                "content": "You are a math problem-solving assistant."
            },
            {
                "role": "user",
                "content": content
            }
        ]
    }

    # 发送请求
    start_time = time.time()
    response = requests.post(url, headers=headers, json=payload)
    end_time = time.time()

    model_answer = "ERROR"
    tokens_generated = 0
    time_elapsed = round(end_time - start_time, 4)
    tokens_per_second = None

    if response.status_code == 200:
        output_data = response.json()
        model_answer = output_data['choices'][0]['message']['content'].strip()
        tokens_generated = output_data.get('usage', {}).get('completion_tokens', 0)
        tokens_per_second = round(tokens_generated / time_elapsed, 2) if time_elapsed > 0 else None
    else:
        print(f"Error {response.status_code} for qa_id {qa_id}: {response.text}")

    results.append({
        "qa_id": qa_id,
        "question": question,
        "model_answer": model_answer,
        "ground_truth": ground_truth,
        "source": source,
        "answer_type": answer_type,
        "problem_type": problem_type,
        "complexity_level": complexity_level,
        "tokens_generated": tokens_generated,
        "time_elapsed": time_elapsed,
        "tokens_per_second": tokens_per_second
    })

    # 保存结果
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
