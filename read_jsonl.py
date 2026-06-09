import json
import jsonlines

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

if __name__ == "__main__":
    file_path = "checkpoints/verl_agent_webshop/webshop_validation/validation_generations/0.jsonl"  # Replace with your JSONL file path
    data = read_jsonl(file_path)
    for item in data:
        print(item.keys())
        breakpoint()