from decision.scenegraph_utils import *


if __name__ == "__main__":
    prompt: str = '''
        You are an AI assistant with commonsense and strong ability to infer the distance between a subgraph
        and a goal in an indoor scene. 

        You need to predict the most likely distance of a subgraph and a goal in a room. You need to answer the distance in 
        meters and give your reason. Here is the JSON format:
        Input:
        {{"subgraph": {{"nodes": ["sofa", "table", ...], "edges": ["sofa next to table", ...]}}, "goal": "TV"}}
        Response: 
        {{"distance": 2, "reason": "Becasue TV and sofa are on both sides of table."}}

        Now predict the distance and give your reason: 
        {{"subgraph": {}, "goal": {}}}
    '''

    subgraph: str = '''{{"nodes": ["wall", "flag", "chair", "dresser"], "edges": ["wall under flag", "flag next to chair", "chair next to dresser"]}}'''
    goal: str = "bed" 

    prompt = prompt.format(subgraph, goal)
    print(prompt)
    distance, reason = get_subgraph_score_from_llm(prompt, "GPT")
    distance = distance.split(' ')[0]
    print(distance, reason)
    assert isinstance(float(distance), float), "GPT provides wrong distance format"
    assert isinstance(reason, str), "GPT provides wrong reason data type"