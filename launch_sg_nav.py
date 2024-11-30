import os
import json
import asyncio
import logging
import argparse
from queue import Queue
from threading import Thread

import yaml

from decision import (
    dict_to_namespace, 
    CognitiveUnit,
    CLIP_LLM_FMMAgent_NonPano
)
from remote import ReceiverPeer
from remote.receiver_peer import DataSynchonizer


def start_distributed_sg_nav(
    agent: CognitiveUnit,
    observation_queue: Queue,
    done: asyncio.Event,
    complete_event: asyncio.Event,
    loop: asyncio.AbstractEventLoop
) -> None:
    while not done.is_set():
        observations: dict = observation_queue.get()
        if observations["reset"]:
            print("Reset signal received...")
            agent.reset()
            continue

        agent.act(observations)
        complete_event.set()

    loop.stop()
    logging.info("Evaluation complete, waiting for finish...")


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
        "--visulize", action='store_true'
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
    core_agent: CLIP_LLM_FMMAgent_NonPano = CLIP_LLM_FMMAgent_NonPano(config, args)
    agent: CognitiveUnit = CognitiveUnit(core_agent)

    with open("ip_configs.json", "r") as f:
        config: dict = json.load(f)
    receiver: ReceiverPeer = ReceiverPeer(config['signaling_ip'], config['port'], config['stun_url'])
    synchronizer: DataSynchonizer = DataSynchonizer()
    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    async_queue_names: list = ["rgb", "depth", "state", "semantic"]
    queue_names: list = ["action", "step"]

    receiver.set_loop(loop)
    synchronizer.set_loop(loop)
    synchronizer.done = receiver.done
    for name in async_queue_names:
        queue = asyncio.Queue(config['max_size'])
        receiver.set_queue(name, queue)
        synchronizer.set_queue(name, queue)
    for name in queue_names:
        queue = Queue(config['max_size'])
        receiver.set_queue(name, queue)
        synchronizer.set_queue(name, queue)
    agent.set_queue('action', receiver.action_queue)

    decision_thread: Thread = Thread(
        target=start_distributed_sg_nav,
        args=(
            agent,
            receiver.step_queue,
            receiver.done,
            receiver.action_event,
            loop
        )
    )

    try:
        loop.create_task(receiver.run())
        loop.create_task(synchronizer.syncronize_to_step())
        decision_thread.start()
        loop.run_forever()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Closing receiver and loop...")
        receiver.stop()

        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

        loop.stop()
        loop.close()
        decision_thread.join()
        print("Program finished")
