import json
import time
import random
import logging
import asyncio
from queue import Queue
from threading import Thread

import cv2
import numpy as np
from remote import ReceiverPeer
from remote.receiver_peer import AsyncAligner
from remote.comm_utils import push_to_buffer


# Note: Only for the integration test on the local machine
def process_step_data(
    step_queue: Queue,
    action_queue: Queue,
    event: asyncio.Event,
    complete_event: asyncio.Event,
    loop: asyncio.AbstractEventLoop
) -> None:
    last_step = None

    while not event.is_set():
        step_data: dict = step_queue.get()
        print(step_data["reset"], step_data["step"])

        if step_data["reset"]:
            last_step = step_data['step']
            print("Reset signal received, resetting...")
            continue

        if last_step != step_data['step']:
            last_step = step_data['step']
            cv2.imwrite(f"RGB.png", step_data["rgb"])
            depth_map_uint8 = (step_data["depth"] * 255).astype(np.uint8)
            cv2.imwrite(f"Depth.png", depth_map_uint8)
            # cv2.imshow("RGB received", step_data["rgb"])
            # cv2.imshow("Depth received", step_data["depth"])
            # cv2.waitKey(30)

            time.sleep(10) # Simulate processing time

            action: dict = {"action": random.randint(0, 5)}
            action_queue.put(action.copy())
            print(f"Putting {action} to the buffer...")
            complete_event.set()

    loop.stop()
    print("Test complete, waiting for finish...")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    with open("ip_configs.json", "r") as f:
        config: dict = json.load(f)

    receiver: ReceiverPeer = ReceiverPeer(config['signaling_ip'], config['port'], config['stun_url'])
    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    async_queue_names: list = [] + ["rgb", "depth", "state", "semantic"]
    queue_names: list = ["action", "step"] # + ["rgb", "depth", "state", "semantic"]

    receiver.set_loop(loop)
    for name in async_queue_names:
        queue = asyncio.Queue(config['max_size'])
        receiver.set_queue(name, queue)
    for name in queue_names:
        queue = Queue(config['max_size'])
        receiver.set_queue(name, queue)

    decision_thread: Thread = Thread(
        target=process_step_data,
        args=(
            receiver.step_queue,
            receiver.action_queue,
            receiver.done,
            receiver.action_event,
            loop
        )
    )

    # synchronizer_thread: Thread = Thread(
    #     target=synchronizer.run
    # )

    try:
        loop.create_task(receiver.run())
        # synchronizer_thread.start()
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
