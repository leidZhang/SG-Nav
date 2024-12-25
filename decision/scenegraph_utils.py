import os
import re
import json
from collections import Counter 
from typing import Tuple, List

from openai import OpenAI


def extract_number(input: str) -> str:
    match = re.search(r"\d+\.\d+|\d+", input)
    if match:
        return match.group()
    return "2.00"

def extract_relations(response: str) -> list:
    raw_data: str = response.replace('\n', '').strip('{}')
    data: list = raw_data.split(',')
    cleaned_data_list: list = []
    for relation in data:
        tokens: list = relation.split(':')
        if len(tokens) == 2:
            cleaned_data: str = tokens[1].strip().replace('"', '')
            cleaned_data_list.append(cleaned_data)
    return cleaned_data_list


def llm(prompt: str, llm_name: str = "GPT") -> str:
    try:
        if llm_name == 'GPT':
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            chat_completion = client.chat.completions.create(  # added by someone
                model="gpt-3.5-turbo",
                # model="gpt-4",  # gpt-4
                messages=[{"role": "user", "content": prompt}],
                timeout=120,  # Timeout in seconds
            )
            return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error due to {e}")
        return ''

def get_subgraph_score_from_llm(prompt: str, llm_name: str) -> Tuple[float, str]:
    response: str = llm(prompt, llm_name)
    if response == '':
        return 2.0, ''
    start_index: int = response.find("{")
    if start_index == -1: 
        print("GPT provides wrong answer!")
        return "2.00", ""

    response = response[start_index:]
    print("GPT response: ", response)
    score_dict: dict = json.loads(response)
    return str(score_dict["distance"]), score_dict["reason"]

def get_relations_from_llm(obj_pairs: List[str], prompt: str, llm_name: str = "GPT") -> list:
    obj_pairs_json: str = "{"
    unit_obj_pair: str = '''{{"object1": "{}", "object2": "{}"}}, '''
    for i in range(0, len(obj_pairs)-1, 2):
        obj_pair_json: str = f'"{i // 2}": ' + unit_obj_pair.format(obj_pairs[i], obj_pairs[i+1])
        obj_pairs_json += obj_pair_json
    prompt = prompt + obj_pairs_json[:-2] + "}"
    response: str = llm(prompt, llm_name)  
    if response == '':
        return []
    return extract_relations(response)
    # start_index: int = response.find("{")
    # response = response[start_index:].replace("\n", "")

    # print(response)
    # return json.loads(response)


def find_modes(dict) -> list:
    if dict == {}:
        return ['object']
    else:
        max_count = max(dict.values())
        return [key for key, val in dict.items() if val == max_count]

# def find_modes(lst):  
#     if len(lst) == 0:
#         return ['object']
#     else:
#         counts = Counter(lst)  
#         max_count = max(counts.values())  
#         modes = [item for item, count in counts.items() if count == max_count]  
#         return modes  
    