"""
We have a dataset in a directory of examples that are in text format, that we would like to process
to create json format of the type:
<Start text example>
[2025-03-02T03:50:19.382Z] Task ID: multiagent_crafting_tripwire_hook_full_plan_missing_stick_oak_planks_depth_0
Prompt:
You are a playful Minecraft...
...


Conversation Begin:

Conversation:
[
  {
    "role": "system",
...


Response:
Great! I'll start by checking what we already have. !inventory


<End text example>
We want to turn it into this format, for each file.
<Json example> 
[
  {
    "instruction": "You are a playful Minecraft...",
    "input": "Conversation Begin: \n \n [ { ...",
    "output": "Great! I'll start by checking what we already have. !inventory"
  },
  ...

The files will be in this directory format: 
logs
    task_1
        example_1
        example_2
        ...
    task_2
    ...

This script will run the processing and create one json text file for this named data_mc.json
"""

import os
import json
import re
import argparse

def extract_conversation_parts(text):
    # Extract instruction (prompt)
    prompt_match = re.search(r'Prompt:\n(.*?)\nConversation:', text, re.DOTALL)
    instruction = prompt_match.group(1).strip() if prompt_match else ""

    # Extract input (conversation)
    conv_match = re.search(r'\nConversation:(.*?)\n\nResponse:', text, re.DOTALL) 
    conversation = conv_match.group(1).strip() if conv_match else ""

    # Extract output (response)
    resp_match = re.search(r'Response:\n(.*?)$', text, re.DOTALL)
    response = resp_match.group(1).strip() if resp_match else ""

    return {
        "instruction": instruction,
        "input": conversation,
        "output": response
    }

def process_log_files(input_dir=None, output_file=None):
    data = []
    
    # Use default paths if not specified
    if input_dir is None:
        input_dir = "data/custom/logs_filtered"
    
    if output_file is None:
        output_file = "data/custom/data_mc_filtered.json"
    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "conversation" not in file_path and "memSaving" not in file_path and "coding" not in file_path:
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    conversation_parts = extract_conversation_parts(text)
                    data.append(conversation_parts)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Processed {len(data)} conversations and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Minecraft text logs to JSON format')
    parser.add_argument('--input_dir', type=str, default="data/custom/logs_filtered", 
                        help='Directory containing log files (default: data/custom/logs_filtered)')
    parser.add_argument('--output_file', type=str, default="data/custom/data_mc_filtered.json", 
                        help='Path to output JSON file (default: data/custom/data_mc_filtered.json)')
    
    args = parser.parse_args()
    
    process_log_files(args.input_dir, args.output_file)
