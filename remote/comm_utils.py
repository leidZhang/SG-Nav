import time
import asyncio
from queue import Queue
from abc import ABC
from typing import Any, Union

import numpy as np
from av import VideoFrame
from aiortc import RTCRtpSender, RTCPeerConnection, RTCRtpCapabilities

RGBA_CHANNELS: int = 4
LABEL_MAP_CHANNELS: int = 1
WEIGHTS: np.ndarray = np.array([100, 10, 1])


class InvalidImageShapeError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class InvalidDataTypeError(Exception):
    def __init__(self, message: str) -> None:
        self.message: str = message


def encode_to_rgba(image: np.ndarray) -> np.ndarray:
    height, width, channel = image.shape
    if image.dtype not in [np.float32, np.int32]:
        raise InvalidDataTypeError("Image must be of type float32 or int32")
    if channel != 1:
        raise InvalidImageShapeError("Image must be single channel")

    return image.view(np.uint8).reshape(height, width, RGBA_CHANNELS)

def decode_to_semantic(image: np.ndarray) -> np.ndarray:
    if image.shape[2] != 3:
        raise InvalidImageShapeError("Input image must have 3 channels")
    if image.dtype != np.uint8:
        raise InvalidDataTypeError("Input image must be of type uint8")

    decoded_rgb: np.ndarray = image.astype(np.float32) / 20
    decoded_rgb = np.round(decoded_rgb)
    decoded_rgb = np.dot(decoded_rgb, WEIGHTS).reshape(decoded_rgb.shape[0], decoded_rgb.shape[1], 1)
    return decoded_rgb.astype(np.int32)

def decode_to_depth(image: np.ndarray) -> np.ndarray:
    if image.shape[2] != 3:
        raise InvalidImageShapeError("Input image must have 3 channels")
    if image.dtype != np.uint8:
        raise InvalidDataTypeError("Input image must be of type uint8")

    # Calculate the mean of the three channels to get the denormalized value
    decoded_uint8: np.ndarray = np.mean(image, axis=2).astype(np.uint8)[..., None]
    # Scale the uint8 value to the range [0, 1]
    return decoded_uint8.astype(np.float32) / 255.0


def force_codec(pc: RTCPeerConnection, sender: RTCRtpSender, forced_codec: str = "video/H264") -> None:
    kind: str = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )


def get_frame_from_buffer(buffer: Queue) -> Any:
    if buffer.empty(): # If buffer is empty, directly return None
        time.sleep(0.001)
        return None
    return buffer.get()

def push_to_buffer(buffer: Queue, data: Any) -> None:
    if buffer.full():
        buffer.get()
    buffer.put(data)

async def push_to_async_buffer(buffer: asyncio.Queue, data: Any) -> None:
    if buffer.full():
        await buffer.get()
    await buffer.put(data)

def empty_queue(queue: Queue) -> None:
    print(f"Emptying queue with {queue.qsize()} elements")
    while queue.qsize() > 0:
        queue.get()
    print(f"Current queue size is {queue.qsize()}")

async def empty_async_queue(queue: asyncio.Queue) -> None:
    print(f"Emptying async queue with {queue.qsize()} elements")
    while not queue.empty():
        await queue.get()
    print(f"Current async queue size is {queue.qsize()}")


class BaseAsyncComponent(ABC):
    def __init__(self):
        self.input_queue: Union[Queue, asyncio.Queue] = None
        self.loop: asyncio.AbstractEventLoop = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def set_input_queue(self, input_queue: Union[Queue, asyncio.Queue]) -> None:
        self.input_queue = input_queue
