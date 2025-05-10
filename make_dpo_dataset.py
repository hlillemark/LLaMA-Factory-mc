import re
import json 
import os
import glob
import random

def analyze_json_file(file_path):
    """
    Analyzes a single JSON file to extract the task outcome.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        str or None: The task outcome string if found, otherwise None.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if "turns" in data:
                for turn in data["turns"]:
                    if turn.get("role") == "system" and "content" in turn:
                        if isinstance(turn["content"], str) and "Task ended with score : " in turn["content"]:
                            if "Task ended with score : 1" in turn["content"]:
                                return 1
                            elif "Task ended with score : 0" in turn["content"]:
                                return 0
                            else:
                                score = float(turn["content"].split(":")[-1].strip())
                                return score
                            
                            
        return None
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in: {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}")
        return None
    
def extract_result(folder_path):
    folder_name = os.path.basename(folder_path)
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    # assert len(json_files) == 2, f"Expected 2 json files in {folder_name}, found {len(json_files)}"

    if not json_files:
        return None
    else: 
        score = None
        curr_score = 0
        for json_file in json_files:
            score = analyze_json_file(json_file)
            if score is not None:
                max_score = max(score, curr_score)
                curr_score = max_score

        return curr_score

def find_successful_task_files(
    directory: str,
) -> list[str]:
    """
    Find all subfolders corresponding to a task that has completed successfully and return. 

    Args:
        directory (str): The directory to search in.

    Returns:
        list[str]: A list of file paths that match the criteria.
    """

    successful_task_files = []
    unsuccessful_task_files = []


     # get all folder paths in the given directory
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            if "bots" in folder_path:
                continue
            score = extract_result(folder_path)
            if score is not None:
                if score > 0:
                    successful_task_files.append(folder_path)
                else:
                    unsuccessful_task_files.append(folder_path)
            else:
                print(f"Could not extract result from {folder_path}")

    return successful_task_files, unsuccessful_task_files

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

def filter_data(subfolder: str):
    """
    Process the logs and create a dataset for SFT.
    Args:
        directory (str): The directory containing the logs.
        output_file (str): The output JSON file path.
    """
    successful_task_dirs, unsuccessful_task_dirs = find_successful_task_files(subfolder)
    successful_task_names = [os.path.basename(task_dir) for task_dir in successful_task_dirs]
    print("Number of successful tasks: ", len(successful_task_names))
    unsuccessful_task_names = [os.path.basename(task_dir) for task_dir in unsuccessful_task_dirs]
    successful_text_files, unsuccessful_text_files = [], []
    dataset = []
    # match successful task files with bots/subfolder 
    # and unsuccessful task files with bots/subfolder
    # and create a d
    trajectory_lengths = []
    bots_folder = os.path.join(subfolder, "bots")
    print(f"Processing {bots_folder}")
    # get all subfolder for bots_folder
    for (root, dirs, files) in os.walk(bots_folder):
        task_name = os.path.basename(root)
        if task_name in successful_task_names:
            successful_task_files = glob.glob(os.path.join(root, "*.txt"))
            successful_text_files.extend(successful_task_files)
            conversations = [file_name for file_name in successful_task_files if "conversation" in file_name]
            length_convo = len(conversations)
            trajectory_lengths.append(length_convo)
        elif task_name in unsuccessful_task_names:
            unsuccessful_task_files = glob.glob(os.path.join(root, "*.txt"))
            unsuccessful_text_files.extend(unsuccessful_task_files)
            conversations = [file_name for file_name in unsuccessful_task_files if "conversation" in file_name]
            length_convo = len(conversations)
            trajectory_lengths.append(length_convo)

    success_dataset = []
    for successful_file in successful_text_files:
        with open(successful_file, 'r') as f:
            successful_text = f.read()
        successful_convo_parts = extract_conversation_parts(successful_text)
        success_dataset.append(successful_convo_parts)
    # Save the successful dataset to a JSON file
    unsuccessful_dataset = []
    for unsuccessful_file in unsuccessful_text_files:
        with open(unsuccessful_file, 'r') as f:
            unsuccessful_text = f.read()
        unsuccessful_convo_parts = extract_conversation_parts(unsuccessful_text)
        unsuccessful_dataset.append(unsuccessful_convo_parts)
    # Save the unsuccessful dataset to a JSON file
    print("number of successful data points: ", len(success_dataset))
    return success_dataset, unsuccessful_dataset, {"success_tasks": len(successful_task_dirs), 
                                                   "fail_tasks": len(unsuccessful_task_dirs), 
                                                   "success_examples": len(success_dataset),
                                                   "fail_examples": len(unsuccessful_dataset), 
                                                   "trajectory_lengths": trajectory_lengths,}

def make_sft_dataset(input_dir: str, 
                     success_output_file: str, 
                     fail_output_file: str, 
                     stats_output_file: str = None):
    # get all subfolder for input_dir
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    success_dataset = []
    fail_dataset = []
    meta_data = {}
    for subfolder in subfolders:
        print(f"Processing {subfolder}")
        success_data, fail_data, sub_meta_data = filter_data(subfolder)
        for key, value in sub_meta_data.items():
            if key in meta_data:
                meta_data[key] += value
            else:
                meta_data[key] = value
    # shuffle datasets
    random.shuffle(success_dataset)
    random.shuffle(fail_dataset)
    # Save the successful dataset to a JSON file
    with open(success_output_file, 'w') as f:
        json.dump(success_dataset, f, indent=4)
    # Save the unsuccessful dataset to a JSON file
    with open(fail_output_file, 'w') as f:
        json.dump(fail_dataset, f, indent=4)

    avg_trajectory_lengths = sum(meta_data["trajectory_lengths"]) / len(meta_data["trajectory_lengths"]) if meta_data["trajectory_lengths"] else 0
    meta_data["average_trajectory_length"] = avg_trajectory_lengths
    # remove trajectory_lengths from meta_data
    del meta_data["trajectory_lengths"]
    print(meta_data)
    if stats_output_file:
        # Save the meta data to a JSON file
        with open(stats_output_file, 'w') as f:
            json.dump(meta_data, f, indent=4)

# def dataset_statistics(input_dir):
#     subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]
#     success_dataset = []
#     fail_dataset = []
#     meta_data = {}
#     for subfolder in subfolders:
#         print(f"Processing {subfolder}")
#         success_data, fail_data, sub_meta_data = data_stats_subfolder(subfolder)
#         for key, value in sub_meta_data.items():
#             if key in meta_data:
#                 meta_data[key] += value
#             else:
#                 meta_data[key] = value
#         success_dataset.extend(success_data)
#         fail_dataset.extend(fail_data)
#     # shuffle datasets
#     random.shuffle(success_dataset)
#     random.shuffle(fail_dataset)
#     # Save the successful dataset to a JSON file
#     with open(success_output_file, 'w') as f:
#         json.dump(success_dataset, f, indent=4)
#     # Save the unsuccessful dataset to a JSON file
#     with open(fail_output_file, 'w') as f:
#         json.dump(fail_dataset, f, indent=4)
#     print(meta_data)

# def data_stats_subfolder(subfolder):
#     """
#     Process the logs and create a dataset for SFT.
#     Args:
#         directory (str): The directory containing the logs.
#         output_file (str): The output JSON file path.
#     """
#     successful_task_dirs, unsuccessful_task_dirs = find_successful_task_files(subfolder)
#     successful_task_names = [os.path.basename(task_dir) for task_dir in successful_task_dirs]
#     print("Number of successful tasks: ", len(successful_task_names))
#     unsuccessful_task_names = [os.path.basename(task_dir) for task_dir in unsuccessful_task_dirs]
#     successful_text_files, unsuccessful_text_files = [], []
#     dataset = []
#     # match successful task files with bots/subfolder 
#     # and unsuccessful task files with bots/subfolder
#     # and create a d
#     bots_folder = os.path.join(subfolder, "bots")
#     print(f"Processing {bots_folder}")
#     trajectory_lengths = []
#     # get all subfolder for bots_folder
#     for (root, dirs, files) in os.walk(bots_folder):
#         task_name = os.path.basename(root)
#         if task_name in successful_task_names:
#             successful_task_files = glob.glob(os.path.join(root, "*.txt"))
#             conversations = [file_name for file_name in successful_task_files if "conversation" in file_name]
#             length_convo = len(conversations)
#             trajectory_lengths.append(length_convo)
#         elif task_name in unsuccessful_task_names:
#             unsuccessful_task_files = glob.glob(os.path.join(root, "*.txt"))
#             conversations = [file_name for file_name in unsuccessful_task_files if "conversation" in file_name]
#             length_convo = len(conversations)
#             trajectory_lengths.append(length_convo)

#     success_dataset = []
#     for successful_file in successful_text_files:
#         with open(successful_file, 'r') as f:
#             successful_text = f.read()
#         successful_convo_parts = extract_conversation_parts(successful_text)
#         success_dataset.append(successful_convo_parts)
#     # Save the successful dataset to a JSON file
#     unsuccessful_dataset = []
#     for unsuccessful_file in unsuccessful_text_files:
#         with open(unsuccessful_file, 'r') as f:
#             unsuccessful_text = f.read()
#         unsuccessful_convo_parts = extract_conversation_parts(unsuccessful_text)
#         unsuccessful_dataset.append(unsuccessful_convo_parts)
#     # Save the unsuccessful dataset to a JSON file
#     print("number of successful data points: ", len(success_dataset))
#     return success_dataset, unsuccessful_dataset, {"success_tasks": len(successful_task_dirs), 
#                                                    "fail_tasks": len(unsuccessful_task_dirs), 
#                                                    "success_examples": len(success_dataset),
#                                                    "fail_examples": len(unsuccessful_dataset), 
#                                                    "average_trajectory_length": trajectory_lengths,}

def make_dpo_subfolder_dataset(subfolder: str):
    """
    Process the logs and create a dataset for DPO.
    Args:
        directory (str): The directory containing the logs.
        output_file (str): The output JSON file path.
    """
    successful_task_dirs, unsuccessful_task_dirs = find_successful_task_files(subfolder)
    successful_task_names = [os.path.basename(task_dir) for task_dir in successful_task_dirs]
    unsuccessful_task_names = [os.path.basename(task_dir) for task_dir in unsuccessful_task_dirs]
    successful_text_files, unsuccessful_text_files = [], []
    dataset = []
    # match successful task files with bots/subfolder 
    # and unsuccessful task files with bots/subfolder
    # and create a d
    bots_folder = os.path.join(subfolder, "bots")
    print(f"Processing {bots_folder}")
    # get all subfolder for bots_folder
    for (root, dirs, files) in os.walk(bots_folder):
        task_name = os.path.basename(root)
        if task_name in successful_task_names:
            successful_task_files = glob.glob(os.path.join(root, "*.txt"))
            # sort the text files by name and remove first 15
            successful_task_files.sort()
            successful_task_files = successful_task_files[15:]
            successful_text_files.extend(successful_task_files)
        elif task_name in unsuccessful_task_names:
            unsuccessful_task_files = glob.glob(os.path.join(root, "*.txt"))
            # sort the text files by name and remove first 15
            unsuccessful_task_files.sort()
            unsuccessful_task_files = unsuccessful_task_files[15:]
            unsuccessful_text_files.extend(unsuccessful_task_files)

    for successful_file, unsuccessful_file in zip(successful_text_files, unsuccessful_text_files):
        with open(successful_file, 'r') as f:
            successful_text = f.read()
        with open(unsuccessful_file, 'r') as f:
            unsuccessful_text = f.read()

        dpo_pair = make_dpo_preference_trajectory_pairs(successful_text, unsuccessful_text)
        dataset.append(dpo_pair)

    return dataset

def make_dpo_dataset(input_dir: str, output_file: str):

    # get all subfolder for input_dir
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    dataset = []
    for subfolder in subfolders:
        print(f"Processing {subfolder}")
        dpo_dataset = make_dpo_subfolder_dataset(subfolder)
        dataset.extend(dpo_dataset)
    # Save the dataset to a JSON file
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)

def make_dpo_preference_trajectory_pairs(successful_text, unsuccessful_text):
    successful_convo_parts = extract_conversation_parts(successful_text)
    unsuccessful_convo_parts = extract_conversation_parts(unsuccessful_text)
    conversations = [{"from": "human", "value": successful_convo_parts["instruction"]}]
    chosen = {"from": "gpt", "value": successful_convo_parts["input"]+successful_convo_parts["output"]}
    rejected = {"from": "gpt", "value": unsuccessful_convo_parts["input"]+unsuccessful_convo_parts["output"]}

    return {
        "conversations": conversations,
        "chosen": chosen, 
        "rejected": rejected,
    }


make_sft_dataset(input_dir="downloaded_data/cooking", 
                success_output_file="data/custom/cooking_sft_success.json", 
                fail_output_file="data/custom/cooking_sft_fail.json", 
                stats_output_file="data/custom/cooking/")
# dataset_statistics(input_dir="downloaded_data/cooking")
    


