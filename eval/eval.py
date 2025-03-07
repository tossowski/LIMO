import json
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
import re
import importlib.util
import os
import argparse
import vllm.envs as envs
import random
import time
from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
import pickle
from math import comb

# envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"

def parse_list(arg):
    return arg.split(',')

def save_completions(completions, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--n_sampling', type=int, default=1, help="n for sampling")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--batch_size", type=int, default=100, help="vllm batch size")

    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument('--start_idx', type=int, default=0, help="data[start:end]")
    parser.add_argument('--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--prompt_type", default="qwen-base", type=str)
    parser.add_argument("--prompt_file_path", default="./prompts", type=str)
    parser.add_argument("--surround_with_messages", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument('--stop', type=parse_list)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dtype", default='auto', type=str)
    parser.add_argument("--completions_save_dir", default='./completions', type=str)
    # parser.add_argument("--use_qwen_check", action="store_true")
    args = parser.parse_args()
    
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy 
    print(f"current stop list: {args.stop}")
    return args

def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def get_three_prompt(prompt_type, data_name):
    file_path = os.path.join(".", "prompts", prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    # 动态导入模块
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")
    
    if hasattr(module, 'few_shot_prompt'):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")
    
    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    return system_prompt, few_shot_prompt, question_format


def infer(args):
    model_name_or_path = args.model_name_or_path
    print(f"current eval model: {model_name_or_path}")
    is_multimodal = 'VL' in model_name_or_path
    print(f"Batch Size is {args.batch_size}")
    
    n_sampling = args.n_sampling
    factor = 1
    for i in range(2, 65):
        if n_sampling % i == 0:
            factor = i
    generation_epoch = n_sampling // factor
    print(f"use n = {factor}, generation epoch is: {generation_epoch}")
    sampling_params = SamplingParams(temperature=args.temperature, 
                                     max_tokens=args.max_tokens, 
                                     n=factor,
                                     top_p=args.top_p,
                                     )
    
    examples = load_data(args.data_name, args.split, args.data_dir)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx:args.end_idx]
    
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-3:])
    out_file_prefix = f'{args.split}_{args.prompt_type}_t{args.temperature}'
    out_file = f'{args.output_dir}/{model_name}/{args.data_name}/{out_file_prefix}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}.jsonl'
    
    
    if os.path.exists(out_file):
        print(f"Completely same name file({out_file}) exist, skip generation, save file and check correct")
        return
    
    os.makedirs(f'{args.output_dir}/{model_name}/{args.data_name}', exist_ok=True)
    os.makedirs(f'{args.completions_save_dir}/{model_name}/{args.data_name}', exist_ok=True)
    
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"
    print(f"available_gpus: {available_gpus}")
    
    if is_multimodal:
        print("Model is multimodal, so using processor instead of tokenizer ...")
        tokenizer = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        if "Qwen_7b_finetune_LIMO_only" in model_name: # Temp fix because I trained with special tokens... oops
            new_tokens = ["<think>", "</think>"]
            tokenizer.tokenizer.add_tokens(new_tokens)
            # model_def.resize_token_embeddings(len(tokenizer.tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    prompt_batch = []
    for example in tqdm(examples, total=len(examples)):
        # parse question and answer
        question = parse_question(example, args.data_name)
        system_prompt, few_shot_prompt, question_format = get_three_prompt(args.prompt_type, args.data_name)
        
        if args.use_few_shot:
            cur_prompt = few_shot_prompt + question_format.format(question=question)
        else:
            cur_prompt = question_format.format(question=question)
        if args.surround_with_messages:
            
            if is_multimodal:
                messages = [  
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{"type": "text", "text": cur_prompt}]}
                ]
                if 'image_folder' in example and example['image_folder'] != None:
                    if type(example['image']) == str:
                        example['image'] = [example['image']]

                    

                    for image in example['image']:
                        messages[1]['content'].append({"type": "image", 
                                                    "image": os.path.join(example['image_folder'], image),
                                                    })

                prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)


                image_data, _ = process_vision_info(messages,
                                                    return_video_kwargs=False)
                
                if image_data:
                    cur_prompt = {
                        "prompt": prompt,
                        "multi_modal_data": {
                            "image": image_data
                        }
                    }
                else:
                    cur_prompt = {"prompt" : prompt}
                

            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": cur_prompt}
                ]
                cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)

                # messages[1] = {"role": "user", "content": [{"type": "image", "image": os.path.join(example['image_folder'], example['image'])},
                #               {"type": "text", "text": cur_prompt}]}
                # print(messages)
            
        prompt_batch.append(cur_prompt)
    print(prompt_batch[0])
    
    llm = LLM(model=model_name_or_path,
              tensor_parallel_size=len(available_gpus), 
              trust_remote_code=True, 
            #   swap_space=60,
              gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": 10, "video": 10}
              )
    
    
    file_outputs = []
    correct_cnt = 0
    for cur_generation_epoch in range(generation_epoch):
        completions_save_file = f'{args.completions_save_dir}/{model_name}/{args.data_name}/{out_file_prefix}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}_gen_round{cur_generation_epoch}.pkl'
        #print(prompt_batch)
        completions = []
        for i in range(0, len(prompt_batch), args.batch_size):
            batch = prompt_batch[i : min(i + args.batch_size, len(prompt_batch))]
            completions.extend(llm.generate(batch, sampling_params))
        #completions = llm.generate(prompt_batch, sampling_params)
        
        save_completions(completions, completions_save_file)
        for i in range(len(examples)):
            d = examples[i]
            question = parse_question(d, args.data_name)
            generated_responses = [completions[i].outputs[j].text for j in range(len(completions[i].outputs))]
            if cur_generation_epoch == 0:
                file_outputs.append({
                    "question": question,
                    "generated_responses": generated_responses,
                })
                if "id" in d:
                    file_outputs[i]["id"] = d["id"]
                if "source" in d:
                    file_outputs[i]["source"] = d["source"]
            else:
                file_outputs[i]['generated_responses'] += generated_responses
    print("llm generate done")
    print(len(file_outputs))
    
    pass_at_k_list = []
    k = args.k
    

    for i in tqdm(range(len(examples)), "check correct..."):
        d = examples[i]
        gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
        generated_responses = file_outputs[i]['generated_responses']
        generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
        is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
        #is_correct_list = [multiple_choice_correct(d['problem'], generated_answer, gt_ans) for generated_answer in generated_answers]

        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1
        file_outputs[i]['generated_answers'] = generated_answers
        file_outputs[i]['gold_answer'] = gt_ans
        file_outputs[i]['is_correct'] = is_correct
        file_outputs[i]['answers_correctness'] = is_correct_list
        
        if len(is_correct_list) > 1:
            correct_answers = sum(is_correct_list)
            n = len(generated_answers)
            if correct_answers > 0:
                if n - correct_answers < k:
                    pass_at_k = 1
                else:
                    pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                pass_at_k_list.append(pass_at_k)
            else:
                pass_at_k_list.append(0)
                

            
    
    temp_out_file = out_file + ".tmp"
    with open(temp_out_file, 'w', encoding='utf-8') as f:
        count = 0
        for d in tqdm(file_outputs, "writing generation to jsonl file..."):
            f.write(json.dumps(d, ensure_ascii=False))
            f.write("\n")
            count += 1
            if count % 100 == 0:
                f.flush()
        f.flush()
    os.rename(temp_out_file, out_file)

    os.makedirs(f"final_results/{model_name_or_path}", exist_ok=True)
    with open(f"final_results/{model_name_or_path}/results.jsonl", "a") as f:
        new_entry = {}
        new_entry['task'] = args.data_name

        if type(prompt_batch[0]) == str:
            new_entry['example_prompt'] = prompt_batch[0]
        else:
            new_entry['example_prompt'] = prompt_batch[0]['prompt']
        print(f"correct cnt / total cnt: {correct_cnt}/{len(examples)}")
        print(f"Acc: {correct_cnt / len(examples):.4f}")
        new_entry['accuracy'] = f"{correct_cnt / len(examples):.4f}"
        

        if pass_at_k_list:
            average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
            print(f"Pass@{k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
            new_entry[f"pass@{k}"] = f"{average_pass_at_k:.4f}"
        else:
            print(f"Pass@1: {correct_cnt}/{len(examples)} = {correct_cnt / len(examples):.4f}")
            new_entry[f"pass@{k}"] = f"{correct_cnt / len(examples):.4f}"
        f.write(json.dumps(new_entry) + "\n")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    infer(args)
