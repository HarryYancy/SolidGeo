import json
import multiprocessing as mp
from datetime import datetime
import os
from openai import OpenAI
from tqdm import tqdm
from functools import partial
import argparse
import pandas as pd

def ensure_output_dir(output_file):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def process_chunk(chunk, api_key, base_url):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    results = []
    for item in chunk:
        try:
            qa_id = item['qa_id']
            question = item['question']
            model_answer = item['result']
            ground_truth = item['answer']
            source = item.get('source', None)
            answer_type = item.get('answer_type', None)
            problem_type = item.get('problem_type', None)
            complexity_level =  item.get('complexity_level', None)
            # if len(model_answer) > 10001:
            #     model_answer = model_answer[:10000]
            prompt_system = """You are a strict evaluator for exam questions.

                            Your task is to determine whether the model's answer fully and correctly matches the standard answer.

                            Rules:
                            - The question may involve multiple steps, sub-questions, or reasoning parts.
                            - The model's answer must be entirely correct in **all steps** to be accepted.
                            - If any single part is incorrect, incomplete, or missing — the answer must be judged as incorrect.
                            - Do not tolerate partially correct answers or reasoning with errors in any step.

                            Return 'True' only if **all steps and results** in the model's answer are fully correct.
                            Otherwise, return 'False'. Do not explain or justify.
                            """
            
            prompt_user = f"""
                            Question: {question}
                            Standard Answer: {ground_truth}
                            Model Answer: {model_answer}

                            Evaluate whether the model's answer is completely correct.

                            If the question involves multiple steps or sub-questions, then **every part must be answered correctly**.
                            If **any single step or sub-question is incorrect**, the whole answer must be considered incorrect.

                            Reply strictly with True or False only.
                            """
            
            messages = [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ]

            response = client.chat.completions.create(
                model="deepseek-v3",
                # model="qwen-max-latest",
                messages=messages,
                max_tokens=10,
                temperature=0.0,
                stream=False
            )

            is_correct = response.choices[0].message.content.strip().lower() == 'true'
            
            results.append({
                "qa_id": qa_id,
                "question": question,
                "model_answer": model_answer,
                "ground_truth": ground_truth,
                "source": source,
                "answer_type": answer_type,
                "problem_type": problem_type,
                "complexity_level": complexity_level,
                "is_correct": is_correct
            })
            
        except Exception as e:
            print(f"Error occurred while processing QA_ID {qa_id}: {str(e)}")
            print(f"This response is treated as False!")
            results.append({
                "qa_id": qa_id,
                "question": question,
                "model_answer": model_answer,
                "ground_truth": ground_truth,
                "source": source,
                "answer_type": answer_type,
                "problem_type": problem_type,
                "complexity_level": complexity_level,
                "is_correct": False
            })
            continue
            
    return results

def statics(results):
    eva = pd.DataFrame(results)

    eva['qa_id'] = eva['qa_id'].astype('int64')

    groups=['source', 'complexity_level', 'problem_type', 'answer_type']
    result={}
    for group in groups:
        if group != 'problem_type':
            # Group and calculate accuracy, total count, and correct count
            accuracy_stats = eva.groupby(group)['is_correct'].agg(['mean', 'count', 'sum']).reset_index()
            accuracy_stats.columns = [group, 'accuracy', 'total_count', 'correct_count']
            result[group]=accuracy_stats.to_dict(orient='records')
        else:
            # Explode the problem_type column
            exploded_data = eva.explode('problem_type')

            # Group and calculate accuracy, total count, and correct count
            accuracy_stats = exploded_data.groupby('problem_type')['is_correct'].agg(
                ['mean', 'count', 'sum']).reset_index()
            accuracy_stats.columns = ['problem_type', 'accuracy', 'total_count', 'correct_count']
            result['problem_type'] = accuracy_stats.to_dict(orient='records')

    return result

def evaluate_answers_parallel(results_file, output_file, num_processes=4, chunk_size=10):
    ensure_output_dir(output_file)
    
    api_key = ""
    base_url = ""
    
    # Read the results file
    with open(results_file, 'r', encoding='utf-8') as f:
        all_items = pd.read_json(f,orient='records')

    results = []

    # Read intermediate results from the output file
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)['results']
        print(f"Intermediate results read from output file, total {len(results)} items")
        
        # Extract all processed qa_id
        processed_qa_ids = {result['qa_id'] for result in results}

        # Filter out processed rows
        all_items = all_items[~all_items['qa_id'].isin(processed_qa_ids)]

    all_items = all_items.to_dict(orient='records')


    # Split data into chunks
    chunks = [all_items[i:i + chunk_size] for i in range(0, len(all_items), chunk_size)]
    
    # Create process pool and progress bar
    with mp.Pool(processes=num_processes) as pool:
        process_func = partial(process_chunk, api_key=api_key, base_url=base_url)

        
        
        # Use tqdm to display progress
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for chunk_result in pool.imap_unordered(process_func, chunks):
                results.extend(chunk_result)
                pbar.update(1)
    
    # Calculate statistics
    total_questions = len(results)
    correct_answers = sum(1 for r in results if r["is_correct"])
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    stats=statics(results)

    # Create final output
    final_output = {
        "results": results,
        "statistics": {
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "accuracy": accuracy
        },
        "detailed_Statistics": stats
    }
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    print(f"\nEvaluation results saved to: {output_file}")
    print(f"Total questions: {total_questions}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=str, default="/path/to/results.json")
    parser.add_argument("--output-file", type=str, default="/path/to/output.json")
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=10)
    args = parser.parse_args()

    evaluate_answers_parallel(
        results_file=args.results_file,
        output_file=args.output_file,
        num_processes=args.num_processes,
        chunk_size=args.chunk_size
    )
