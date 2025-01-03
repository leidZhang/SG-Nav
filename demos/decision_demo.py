import os
import argparse
from types import SimpleNamespace

import yaml
import numpy as np

from decision.glip_utils import *
from decision import CLIP_LLM_FMMAgent_NonPano


def dict_to_namespace(d):
    # Recursively convert dicts to SimpleNamespace
    return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", default="local", type=str, choices=["local", "remote"]
    )
    parser.add_argument(
        "--PSL_infer", default="one_hot", type=str, choices=["optim", "one_hot"]
    )
    parser.add_argument(
        "--reasoning", default="both", type=str, choices=["both", "room", "obj"]
    )
    parser.add_argument(
        "--llm", default="deberta", type=str, choices=["deberta", "chatgpt"]
    )
    parser.add_argument(
        "--error_analysis", default=False, type=bool, choices=[False, True]
    )
    parser.add_argument(
        "--visulize", default=True,
    )
    parser.add_argument(
        "--split_l", default=0, type=int
    )
    parser.add_argument(
        "--split_r", default=11, type=int
    )
    args = parser.parse_args()

    config_path = "configs/objectnav_hm3d.yaml"
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    config = dict_to_namespace(config_dict)
    agent = CLIP_LLM_FMMAgent_NonPano(config, args)


    test_data: dict = np.load("demos/test_data.npz", allow_pickle=True)
    keys = test_data.files
    print(keys)
    for i in range(1):
        agent.reset()        
        for j in range(len(test_data[keys[0]])):
            observation = {}
            for key in keys:
                if key == "semantic":
                    continue

                observation[key] = test_data[key][j]

            # observation['depth'] = observation['depth'] # * 4.5 + 0.5
            # observation['depth'] = observation['depth'].reshape((480, 640))
            action: dict = agent.act(observation)
            print(action)

    # for i in range(100):
    #     for observations in test_data["data"]:
    #         action: dict = agent.act(observations)
    #         print(action)
    # print("Test complete")