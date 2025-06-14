import json
from datasets import Dataset, DatasetDict

input_file_path = "data/custom/dpo_pairs_crafting.json"
output_file_path = "data/custom/dpo_pairs_crafting_filtered.json"
output_file_path_llama_factory = "data/custom/dpo_pairs_crafting_filtered_llama_factory.json"

def filter_dpo_data(input_file_path, output_file_path):

    with open(input_file_path, "r") as file:
        data = json.load(file)

    new_data = []
    for example in data: 
        if example["preferred"] == "":
            continue
        if example["preferred"] == example["dispreferred"]:
            continue
        if "Hello World" in example["preferred"]:
            continue
        # parse input as a list
        conversation = json.loads(example["input"])
        systemMessage = example["instruction"]
        lst = [{ 'role': 'system', 'content': systemMessage }]
        lst.extend(conversation)

        chosen_lst = lst + [{
            'role': 'assistant',
            'content': example["preferred"]
        }]
        rejected_lst = lst + [{
            'role': 'assistant',
            'content': example["dispreferred"]
        }]

        new_example = {
            "chosen": chosen_lst,
            "rejected": rejected_lst
        }

        new_data.append(new_example)
    
    print("Filtered data length:", len(new_data))
    dataset = Dataset.from_list(new_data)
    dataset.push_to_hub("izzcw/dpo_pairs_crafting_filtered")
    with open(output_file_path, "w") as file:
        json.dump(new_data, file, indent=4)

def filter_dpo_data_llama_factory(input_file_path, output_file_path):
    new_data = []

    with open(input_file_path, "r") as file:
        data = json.load(file)

    for example in data:
        if example["preferred"] == "":
            continue
        if example["preferred"] == example["dispreferred"]:
            continue
        if "Hello World" in example["preferred"]:
            continue

        conversation = json.loads(example["input"])
        systemMessage = example["instruction"]
        lst = [{ 'role': 'system', 'content': systemMessage }]
        lst.extend(conversation)
        
        new_example = {
            "conversations": lst,
            "chosen": {
                "from": "assistant",
                "value": example["preferred"]
            },
            "rejected": {
                "from": "assistant",
                "value": example["dispreferred"]
            }
        }

        new_data.append(new_example)

    print("Filtered data length:", len(new_data))
    with open(output_file_path, "w") as file:
        json.dump(new_data, file, indent=4)

if __name__ == "__main__":
    # filter_dpo_data(input_file_path, output_file_path)
    filter_dpo_data_llama_factory(input_file_path, output_file_path_llama_factory)
    print(f"Filtered data saved to {output_file_path}")