import json

from config import client
from models import CHAT_MODEL

def extract_memory(query):
    prompt = f"""Extract important, persistent, user-specific information from the given query.
    Ignore general knowledge and any information that is not relevant to the user's long-term memory. 
    Focus on details that can be stored and recalled later.
    Do not infer or invent any information on your own.
    Return the extracted information in a structured JSON format, with keys representing the type of information and values representing the details.
    Return ONLY valid JSON. Do not include markdown code fences, explanations, or any text outside the JSON object.
    The query is as follows: {query}"""
    
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )
    response_text = response.text

    response_text = response_text.replace("```json", "")
    response_text = response_text.replace("```", "")

    try:
        memory = json.loads(response_text)
        return memory

    except json.JSONDecodeError:
        return {}

def save_long_term_memory(memory):
    existing_memory = load_long_term_memory()

    existing_memory.update(memory)

    with open("long_term_memory.json", "w") as f:
        json.dump(existing_memory, f, indent=4)


def load_long_term_memory():
    try:
        with open("long_term_memory.json", "r") as f:
            data = json.load(f)
        return data

    except (json.JSONDecodeError, FileNotFoundError):
        return {}
    
    
    