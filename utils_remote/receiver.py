import asyncio
import json
import base64
import logging

import cv2
import numpy as np
from aiortc import RTCPeerConnection
from aiortc.contrib.signaling import TcpSocketSignaling


def decode_image(stringfied_image: str) -> np.ndarray:
    image_data = base64.b64decode(stringfied_image)
    buffer = np.frombuffer(image_data, np.int8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


async def receive_dict() -> None:
    import warnings
    warnings.warn("receive_dict method is no longer maintained and will be removed in the future")

    logging.basicConfig(level=logging.INFO)
    pc: RTCPeerConnection = RTCPeerConnection()

    @pc.on('datachannel')
    def on_datachannel(channel):
        @channel.on('message')
        def on_message(message):
            data: dict = json.loads(message)
            image: np.ndarray = decode_image(data['image'])
            pose: list = data["pose"]

            print(f"Received pose {pose}")
            cv2.imshow('Received Image', image)
            cv2.waitKey(30)

    signaling: TcpSocketSignaling = TcpSocketSignaling('localhost', 1234)
    await signaling.connect()
    logging.info("Connected to signaling server")

    offer = await signaling.receive()
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await signaling.send(pc.localDescription)
    logging.info("Answer sent to signaling server")
    logging.info("Receiver shut down in 10 seconds...")

    await asyncio.sleep(10)  


if __name__ == '__main__':
    asyncio.run(receive_dict())
