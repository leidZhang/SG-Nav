import os
import json
from typing import Tuple, List

from openai import OpenAI


def llm(prompt: str, llm_name: str = "GPT") -> str:
    if llm_name == 'GPT':
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            chat_completion = client.chat.completions.create(  # added by someone
                model="gpt-3.5-turbo",
                # model="gpt-4",  # gpt-4
                messages=[{"role": "user", "content": prompt}],
                # timeout=10,  # Timeout in seconds
            )
            return chat_completion.choices[0].message.content
        except:
            return ''

def get_subgraph_score_from_llm(prompt: str, llm_name: str) -> Tuple[float, str]:
    response: str = llm(prompt, llm_name)
    start_index: int = response.find("{")
    response = response[start_index:]
    score_dict: dict = json.loads(response)
    return score_dict["distance"], score_dict["reason"]

def get_relations_from_llm(obj_pairs: List[str], prompt: str, llm_name: str = "GPT") -> dict:
    obj_pairs_json: str = "{"
    unit_obj_pair: str = '''{{"object1": "{}", "object2": "{}"}}, '''
    for i in range(0, len(obj_pairs)-1, 2):
        obj_pair_json: str = f'"{i // 2}": ' + unit_obj_pair.format(obj_pairs[i], obj_pairs[i+1])
        obj_pairs_json += obj_pair_json
    prompt = prompt + obj_pairs_json[:-2] + "}"

    response: str = llm(prompt, llm_name)
    start_index: int = response.find("{")
    response = response[start_index:]
    print(response)
    return json.loads(response)
