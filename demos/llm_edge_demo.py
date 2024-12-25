import json

from decision.scenegraph_utils import get_relations_from_llm, extract_relations


if __name__ == "__main__":
    prompt: str = '''
        You are an AI assistant with commonsense and strong ability to infer the spatical
        relationships in a indoor scene.
        You need to provide a spatial relationship between the several pairs of objects. Relationships
        include "next to", "above", "opposite to", "below", "inside", "behind", "in front of"
        All the pairs of objects are provided in JSON fromat, and you also need to response to each
        pair in JSON format and with the same order. Here are 2 examples:
        1.
        Input:
        {"0": {"object1": "chair", "object2": "table"}, "1": {"object1": "monitor", "object2": "desk"}
        Response:
        {"0": "next to", "1": "above"}
        2.
        Input:
        {"0": {"object1": "sofa", "object2": "TV"}, "1": {"object1": "plant", "object2": "chair"}}
        Response:
        {"0": "opposite to", "1": "behind"}
        Now you predict the spatial relationship between these pairs of objects:
    ''' 

    demo_obj_pairs: tuple = (
        "TV", "table", "monitor", "desk",
        "chair", "TV", "chair", "monitor",
        "plant", "TV", "plant", "chair",
        "cabinet", "bed", "desk", "bed",
    )

    lst, relations = [], get_relations_from_llm(demo_obj_pairs, prompt, "GPT")
    # for key, val in relations.items():
    #     lst.append(val)

    # relations = extract_relations(response)
    print(relations) # view the output
    assert len(relations) == len(demo_obj_pairs) / 2, "GPT provides the wrong response!"