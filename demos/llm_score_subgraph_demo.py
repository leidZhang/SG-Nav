from decision.scenegraph_utils import *


if __name__ == "__main__":
    prompt: str = '''
        You are an AI assistant with commonsense and strong ability to infer the distance between two objects in an indoor
        scene. 

        You need to predict the most likely distance of two objects in a room. You need to answer the distance in meters and give
        your reason. Here are 1 example:
        Input:
        {{"object1": "table", "object2": "chair"}}
        Response:
        {{"distance": 0.5, "reason": "because there is always a chair next to the table."}}

        Now predict the distance and give your reason: {{"object1": {}, "object2": {}}}
    ''' 

    object1, object2 = "plant", "TV"

    prompt = prompt.format(object1, object2)
    distance, reason = get_subgraph_score_from_llm(prompt, "GPT")
    print(distance, reason)
    assert isinstance(reason, str), "GPT provides wrong reason data type"