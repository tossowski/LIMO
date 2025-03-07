import argparse
import json
import os
import shutil

from utils.utils import save_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_data', type=str, default="/mnt/hanoverdev/scratch/t-tossowski/llava_med_2/data/MMMU_PRO_MEDICINE", help="path to folder containing json file")
parser.add_argument('--split', type=str,  default='test', help="name of json file")

args = parser.parse_args()

data = json.load(open(os.path.join(args.path_to_data, f"{args.split}.json")))
all_entries = []

for entry in data:
    new_entry = {}
    new_entry['id'] = entry['id']
    new_entry['problem'] = entry['conversations'][0]['value'].replace("<image>\n", "")
    if 'image_folder' in entry:
        new_entry['image_folder'] = entry['image_folder']
        new_entry['image'] = entry['image']
    new_entry['answer'] = entry['conversations'][1]['value'][0]
    all_entries.append(new_entry)

new_folder_name = args.path_to_data.split("/")[-1]
os.makedirs(os.path.join("./data", new_folder_name), exist_ok=True)
save_jsonl(all_entries, os.path.join("./data", new_folder_name, "test.jsonl"))

# Copy prompt template
file_path = "./prompts/qwen-instruct/aime.py"

# New file name
new_file_path = os.path.join(os.path.dirname(file_path), new_folder_name + ".py")

# Copy the file
shutil.copy(file_path, new_file_path)

print(f"Copied template to {new_file_path}")













