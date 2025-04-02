# Check if the task was successful
import os
import json
import re

def check_task_success(root_dir, exp_dir):
    is_successful = False
    score_found = False
    full_exp_path = os.path.join(root_dir, exp_dir)
    
    # Get all JSON files in the experiment directory
    agent_files = [f for f in os.listdir(full_exp_path) if f.endswith(".json")]
    
    # Check each agent file for success information
    for agent_file in agent_files:
        agent_file_path = os.path.join(full_exp_path, agent_file)
        
        try:
            with open(agent_file_path, 'r') as f:
                agent_data = json.load(f)
                
            # Check for score information in the turns data
            if "turns" in agent_data:
                for turn in agent_data["turns"]:
                    if turn.get("role") == "system" and "content" in turn:
                        if isinstance(turn["content"], str) and "Task ended with score : " in turn["content"]:
                            score_found = True
                            if "Task ended with score : 1" in turn["content"]:
                                is_successful = True
                                break
            
            # If we found success, no need to check other files
            if is_successful:
                return is_successful
        except Exception as e:
            print(f"Error loading file {agent_file_path}: {e}")

def load_files(root_dir):
    # Get all experiment directories
    exp_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # Check each experiment directory for success
    successful_dirs = []
    unsuccessful_dirs = []
    for exp_dir in exp_dirs:
        is_successful = check_task_success(root_dir, exp_dir)
        if is_successful:
            successful_dirs.append(exp_dir)
        else:
            unsuccessful_dirs.append(exp_dir)
        print(f"Experiment {exp_dir} successful: {is_successful}")
    return successful_dirs, unsuccessful_dirs

def files_to_data(root_dir, exp_dirs):
    for exp_dir in exp_dirs:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        conversation_parts = extract_conversation_parts(text)
                        data.append(conversation_parts)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    

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

def convert_convo_parts_to_kto(convo, label):
    kto_convo = []
    for item in convo:
        new_items  = {}
        messages = []
        messages.append({
            "content": item['instruction'],
            "role": "system"
        })
        inp = json.loads(item["input"])
        for interaction in inp:
            messages.append({
                "content": interaction["content"],
                "role": interaction["role"]
            })
        out = item["output"]
        messages.append({
            "content": out,
            "role": "assistant"
        })
        new_items["messages"] = messages
        new_items["label"] = label
        kto_convo.append(new_items)
    return kto_convo

def process_log_files(root_dir):
    successful_dirs, unsuccessful_dirs = load_files(root_dir)


