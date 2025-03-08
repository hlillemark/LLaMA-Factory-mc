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

def process_log_files():
    data = []
    base_logs_dir = "data/custom/logs_filtered"

    # Walk through all subdirectories
    for chunk in os.listdir(base_logs_dir):
        chunk_dir = os.path.join(base_logs_dir, chunk)
        for task_dir in os.listdir(chunk_dir):
            task_path = os.path.join(chunk_dir, task_dir)
            if not os.path.isdir(task_path):
                continue

            # Process each example file in task directory
            for example_file in os.listdir(task_path):
                file_path = os.path.join(task_path, example_file)
                if not os.path.isfile(file_path):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        conversation_parts = extract_conversation_parts(text)
                        data.append(conversation_parts)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Write output JSON file
    with open('data/custom/data_mc_filtered.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    process_log_files()
