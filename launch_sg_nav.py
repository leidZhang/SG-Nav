import os
import json
import asyncio
import logging
import argparse
from queue import Queue
from threading import Thread
from typing import List
from multiprocessing import Process
from multiprocessing import Queue as MpQueue

import yaml

from decision import (
    dict_to_namespace, 
    CognitiveUnit,
    CLIP_LLM_FMMAgent_NonPano
)
from remote import ReceiverPeer


def start_distributed_sg_nav(
    comm_process: Thread,
    config: dict, 
    args: dict,
    observation_queue: MpQueue,
    action_queue: MpQueue
) -> None:
    last_step = None

    core_agent: CLIP_LLM_FMMAgent_NonPano = CLIP_LLM_FMMAgent_NonPano(config, args)
    agent: CognitiveUnit = CognitiveUnit(core_agent)
    agent.set_queue('action', action_queue)
    comm_process.start()

    while True:
        observations: dict = observation_queue.get()
        if observations["reset"]:
            print("Reset signal received...")
            agent.reset()
            continue

        if last_step != observations['step']:
            print("Executing step", observations['step'])
            last_step = observations['step']
            agent.act(observations)


def start_aiortc_peer(
    loop: asyncio.AbstractEventLoop,
    peer_config: dict,
    mt_queue_names: List[str], 
    mp_queue_names: List[str],
    mp_queue_list: List[MpQueue]
) -> None:
    receiver: ReceiverPeer = ReceiverPeer(peer_config['signaling_ip'], peer_config['port'], peer_config['stun_url'])

    receiver.set_loop(loop)
    # synchronizer.done = receiver.done
    for i, name in enumerate(mp_queue_names):
        queue = MpQueue(peer_config['max_size'])
        receiver.set_queue(name, mp_queue_list[i])
        mp_queue_list.append(queue)

    for name in mt_queue_names:
        queue = asyncio.Queue(peer_config['max_size'])
        receiver.set_queue(name, queue)

    loop.create_task(receiver.run())
    # synchronizer_thread.start()
    loop.run_forever()


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
        "--llm", default="chatgpt", type=str, choices=["deberta", "chatgpt"]
    ) # deberta
    parser.add_argument(
        "--error_analysis", default=False, type=bool, choices=[False, True]
    )
    parser.add_argument(
        "--visulize", default=False, # action='store_true'
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
    config: dict = dict_to_namespace(config_dict)
    with open("ip_configs.json", "r") as f:
        peer_config: dict = json.load(f)

    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    mp_queue_names: list = ["action", "step"]
    mp_queue_list: list = []    
    mt_queue_names: list = ["rgb", "depth", "state", "semantic"]
    for queue in mp_queue_names:
        mp_queue_list.append(MpQueue(peer_config['max_size']))

    comm_process: Process = Process(
        target=start_aiortc_peer,
        args = (
            loop,
            peer_config,
            mt_queue_names,
            mp_queue_names,
            mp_queue_list
        )
    )

    # sg_nav_process: Thread = Thread(
    #     target=start_distributed_sg_nav,
    #     args=(
    #         config,
    #         args,
    #         mp_queue_list[1],
    #         mp_queue_list[0]
    #     )
    # )

    # comm_process.start()
    # sg_nav_process.start()
    # start_aiortc_peer(loop, peer_config, mt_queue_names, mp_queue_names, mp_queue_list)
    start_distributed_sg_nav(comm_process, config, args, mp_queue_list[1], mp_queue_list[0])

