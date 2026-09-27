import json

def save_memory(history):
    with open("memory.json", "w") as f:
        json.dump(history, f, indent=4)


def load_memory():
    
    try:
        with open("memory.json", "r") as f:
            data = json.load(f)
        return data

    except (json.JSONDecodeError, FileNotFoundError):
        return []
