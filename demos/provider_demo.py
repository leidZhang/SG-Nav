import time
import json
import logging
import asyncio
from queue import Queue
from datetime import datetime
from threading import Thread, Event
from typing import List, Dict

import cv2
import numpy as np

from remote import ProviderPeer
from remote.comm_utils import empty_queue
from decision.agent import HabitatActuator


# TODO: Replace it with real habitat env
def start_habitat(
    agent: HabitatActuator,
    provider_event: asyncio.Event,
    width: int = 640,
    height: int = 480
) -> None:
    print("initializing habitat env...")
    data: Dict[str, List[np.ndarray]] = np.load("demos/test_data.npz", allow_pickle=True)
    mock_observations: List[Dict[str, np.ndarray]] = []
    for i in range(len(data['depth'])):
        observation: Dict[str, np.ndarray] = {}
        for key in data.keys():
            observation[key] = data[key][i]
        mock_observations.append(observation)

    observation = mock_observations[0]
    agent.reset()
    i = 0
    while i < 1000:
        # make sure not stuck here forever
        if provider_event.is_set():
            print("Detected provider stop event, exiting...")
            break

        # get action from agent
        action = agent.act(observation)
        print(f"Got {action} and send {(i + 1) % 256} to the peer...")
        i = (i + 1) % 10 # infinite loop for testing
        observation = mock_observations[i]
        print(f"Received action {action}")
        print("==========================")

    if not provider_event.is_set():
        provider_event.set() # signal provider to stop
    # loop.stop()
    print("Test complete, waiting for finish...")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    with open("ip_configs.json", "r") as f:
        config: dict = json.load(f)

    provider: ProviderPeer = ProviderPeer(config['signaling_ip'], config['port'], config['stun_url'])
    agent: HabitatActuator = HabitatActuator()
    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    queue_names: list = ["depth", "rgb", "state", "action", "semantic"]
    queue_list: list = []

    provider.set_loop(loop)
    for name in queue_names:
        named_queue: Queue = Queue(config['max_size'])
        provider.set_queue(name, named_queue)
        agent.set_queue(name, named_queue)
        queue_list.append(named_queue)
    actuator_thread: Thread = Thread(
        target=start_habitat,
        args=(agent, provider.done, config['width'], config['height'])
    )

    try:
        actuator_thread.start()
        loop.run_until_complete(provider.run())
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Clossing provider and loop...")
        provider.stop()

        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

        loop.stop()
        loop.close()
        actuator_thread.join()
        print("Program finished")