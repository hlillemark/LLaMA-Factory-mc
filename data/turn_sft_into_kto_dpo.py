import json

FILE_PATH = 'data/custom/filtered_cooking_train_data.json'

with open(FILE_PATH, 'r') as f:
    data = json.load(f)
    parse_in_data(data, 'true')

def parse_in_data(data, label):
    for item in data:
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
        new_items["label"] = "true"
        
       