import time
import json
import random
import logging
import asyncio
import fractions
from queue import Queue
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from av import VideoFrame
from aiortc.contrib.media import MediaBlackhole
from aiortc import RTCDataChannel, VideoStreamTrack, RTCRtpSender

from .signaling_utils import WebRTCClient, receive_signaling
from .comm_utils import *

GARBAGE_FRAME: VideoFrame = VideoFrame.from_ndarray(np.zeros((2, 2, 3), dtype=np.uint8), format="rgb24")
GARBAGE_FRAME.pts = 0
GARBAGE_FRAME.time_base = fractions.Fraction(1, 90000)
# Copied from aiortc source code
VIDEO_PTIME = 1 / 10
VIDEO_CLOCK_RATE = 90000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
ASYNC_QUEUE_NAMES: List[str] = [] # ['rgb', 'depth','semantic','state']
QUEUE_NAMES: List[str] = ['action', 'step'] + ['rgb', 'depth','semantic','state']


class RGBProcessor(VideoStreamTrack, BaseAsyncComponent):
    def __init__(self, track: VideoStreamTrack) -> None:
        super().__init__()
        self.track: VideoStreamTrack = track

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if hasattr(self, "_timestamp"):
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, VIDEO_TIME_BASE

    async def recv(self) -> VideoFrame:
        await self.next_timestamp()
        frame: VideoFrame = await asyncio.wait_for(self.track.recv(), timeout=2)   # 2 seconds timeout
        image: np.ndarray = frame.to_ndarray(format="rgb24")

        await self.input_queue.put({'rgb': image, 'pts': frame.pts})
        # await self.loop.run_in_executor(None, self.input_queue.put, {'rgb': image, 'pts': frame.pts})
        # await push_to_async_buffer(self.input_queue, {'rgb': image, 'pts': frame.pts})
        print(f"Received RGB frame at {frame.pts}")
        return GARBAGE_FRAME


class DepthProcessor(VideoStreamTrack, BaseAsyncComponent):
    def __init__(self, track: VideoStreamTrack) -> None:
        super().__init__()
        self.track: VideoStreamTrack = track

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if hasattr(self, "_timestamp"):
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, VIDEO_TIME_BASE

    async def recv(self) -> VideoFrame:
        await self.next_timestamp()
        frame: VideoFrame = await asyncio.wait_for(self.track.recv(), timeout=2) # 2 seconds timeout
        image: np.ndarray = frame.to_ndarray(format="rgb24")
        image = decode_to_depth(image)

        await self.input_queue.put({'depth': image, 'pts': frame.pts})
        # await self.loop.run_in_executor(None, self.input_queue.put, {'depth': image, 'pts': frame.pts})
        # await push_to_async_buffer(self.input_queue, {'depth': image, 'pts': frame.pts})


class SemanticProcessor(VideoStreamTrack, BaseAsyncComponent):
    def __init__(self, track: VideoStreamTrack) -> None:
        super().__init__()
        self.track: VideoStreamTrack = track

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if hasattr(self, "_timestamp"):
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, VIDEO_TIME_BASE

    async def recv(self) -> VideoFrame:
        await self.next_timestamp()
        frame: VideoFrame = await asyncio.wait_for(self.track.recv(), timeout=2) # 2 seconds timeout
        image: np.ndarray = frame.to_ndarray(format="rgb24")
        image = decode_to_semantic(image)

        await self.input_queue.put({'semantic': image, 'pts': frame.pts})
        # await self.loop.run_in_executor(None, self.input_queue.put, {'semantic': image, 'pts': frame.pts})
        # await push_to_async_buffer(self.input_queue, {'semantic': image, 'pts': frame.pts})


class DataSynchronizer:
    def __init__(self, done: asyncio.Event, action_event: asyncio.Event) -> None:
        self.action_event: asyncio.Event = action_event
        self.last_step: int = None
        self.done: asyncio.Event = done
        self.loop: asyncio.AbstractEventLoop = None
        # Queues for each stream/track
        self.depth_queue: asyncio.Queue = None
        self.rgb_queue: asyncio.Queue = None
        self.semantic_queue: asyncio.Queue = None
        self.state_queue: asyncio.Queue = None
        self.step_queue: Queue = None
        self.action_queue: Queue = None

    def run(self) -> None:
        while not self.done.is_set():
            self.__synchronize_to_step()

    def __put_to_step_queue(self, step_data: Dict[str, Any]) -> None:
        # push_to_buffer(self.step_queue, step_data)
        if not self.step_queue.full():
            self.step_queue.put(step_data.copy())

    def __get_latest_data(self, queue: Queue, target_pts: int) -> Dict[str, Any]:
        while True:
            data = queue.get()
            if data['pts'] >= target_pts:
                return data

    def __synchronize_to_step(self) -> None:
        state: Dict[str, Any] = self.state_queue.get()
        rgb_data: Dict[str, Any] = self.rgb_queue.get()
        depth_data: Dict[str, Any] = self.depth_queue.get()
        semantic_data: Dict[str, Any] = self.semantic_queue.get()

        print(rgb_data['pts'], depth_data['pts'], semantic_data['pts'], state['pts'])

        if state["reset"]:
            push_to_buffer(self.step_queue, state.copy())
            return

        max_pts = max(rgb_data['pts'], depth_data['pts'], semantic_data['pts'], state['pts'])
        if rgb_data['pts'] < max_pts:
            rgb_data = self.__get_latest_data(self.rgb_queue, max_pts)
        if depth_data['pts'] < max_pts:
            depth_data = self.__get_latest_data(self.depth_queue, max_pts)
        if semantic_data['pts'] < max_pts:
            semantic_data = self.__get_latest_data(self.semantic_queue, max_pts)
        if state['pts'] < max_pts:
            state = self.__get_latest_data(self.state_queue, max_pts)

        if self.__is_same_step(rgb_data, depth_data, semantic_data, state):
            # print(rgb_data['pts'], depth_data['pts'], semantic_data['pts'], state['pts'])
            step_data: Dict[str, Any] = {
                'rgb': rgb_data['rgb'],
                'depth': depth_data['depth'],
                'semantic': semantic_data['semantic'],
            }
            # print(state) # Print the state data for debugging
            step_data.update(state) # Merge the state with the step data
            self.__put_to_step_queue(step_data)

    def __is_same_step(
        self,
        rgb_data: Dict[str, Any],
        depth_data: Dict[str, Any],
        semantic_data: Dict[str, Any],
        state: Dict[str, Any]
    ) -> bool:
        return (
            rgb_data['pts'] == depth_data['pts'] == semantic_data['pts'] == state['pts']
        )

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def set_queue(self, queue_name: str, queue: Queue) -> None:
        setattr(self, f"{queue_name}_queue", queue)


class AsyncAligner(BaseAsyncComponent):
    def __init__(self, done: asyncio.Event, action_event: asyncio.Event) -> None:
        self.action_event: asyncio.Event = action_event
        self.last_step: int = None
        self.done: asyncio.Event = done
        self.loop: asyncio.AbstractEventLoop = None
        # Queues for each stream/track
        self.depth_queue: asyncio.Queue = None
        self.rgb_queue: asyncio.Queue = None
        self.semantic_queue: asyncio.Queue = None
        self.state_queue: asyncio.Queue = None
        self.step_queue: Queue = None
        self.action_queue: Queue = None

    async def run(self) -> None:
        while not self.done.is_set():
            await self.__synchronize_to_step()

    async def __get_latest_data(self, queue: asyncio.Queue, target_pts: int) -> Dict[str, Any]:
        while True:
            data = await queue.get()
            if data['pts'] >= target_pts:
                return data

    async def __synchronize_to_step(self) -> None:
        state: Dict[str, Any] = await self.state_queue.get()
        if state["reset"]:
            self.last_step = step_data['step']
            await self.loop.run_in_executor(None, self.step_queue.put, state.copy())
            return

        rgb_data: Dict[str, Any] = await self.rgb_queue.get()
        depth_data: Dict[str, Any] = await self.depth_queue.get()
        semantic_data: Dict[str, Any] = await self.semantic_queue.get()

        max_pts = max(rgb_data['pts'], depth_data['pts'], semantic_data['pts'], state['pts'])
        if rgb_data['pts'] < max_pts:
            rgb_data = await self.__get_latest_data(self.rgb_queue, max_pts)
        if depth_data['pts'] < max_pts:
            depth_data = await self.__get_latest_data(self.depth_queue, max_pts)
        if semantic_data['pts'] < max_pts:
            semantic_data = await self.__get_latest_data(self.semantic_queue, max_pts)
        if state['pts'] < max_pts:
            state = await self.__get_latest_data(self.state_queue, max_pts)

        if self.__is_same_step(rgb_data, depth_data, semantic_data, state):
            step_data: Dict[str, Any] = {
                'rgb': rgb_data['rgb'],
                'depth': depth_data['depth'],
                'semantic': semantic_data['semantic'],
            }
            step_data.update(state) # Merge the state with the step data

        if step_data['step'] != self.last_step:
            self.last_step = step_data['step']
            print("Pushing step data to the buffer...")
            await self.loop.run_in_executor(None, self.step_queue.put, step_data.copy())

    def __is_same_step(
        self,
        rgb_data: Dict[str, Any],
        depth_data: Dict[str, Any],
        semantic_data: Dict[str, Any],
        state: Dict[str, Any]
    ) -> bool:
        return (
            rgb_data['pts'] == depth_data['pts'] == semantic_data['pts'] == state['pts']
        )

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def set_queue(self, queue_name: str, queue: Queue) -> None:
        setattr(self, f"{queue_name}_queue", queue)


class ReceiverPeer(WebRTCClient):
    def __init__(
        self,
        signaling_ip: str,
        signaling_port: int,
        stun_urls: List[str] = None
    ) -> None:
        super().__init__(signaling_ip, signaling_port, stun_urls=stun_urls)
        self.data_channel: RTCDataChannel = None
        self.media_processor: MediaBlackhole = None
        self.track_counter: int = 0
        self.action_event: asyncio.Event = asyncio.Event()

        self.loop: asyncio.AbstractEventLoop = None
        # Queues for each stream/track
        self.depth_queue: asyncio.Queue = None
        self.rgb_queue: asyncio.Queue = None
        self.semantic_queue: asyncio.Queue = None
        self.state_queue: asyncio.Queue = None
        self.step_queue: Queue = None
        self.action_queue: Queue = None

    async def send_action(self) -> None:
        while not self.done.is_set():
            action: Dict[str, Any] = await self.loop.run_in_executor(None, self.action_queue.get)
            self.data_channel.send(json.dumps(action))

    async def run(self) -> None: # asyncio.run(receiver.run())
        while not self.disconnected.set():
            await super().run()
            self.__setup_track_callbacks()
            self.__setup_datachannel_callbacks()
            await receive_signaling(self.pc, self.signaling)

            await self.done.wait()
            await self.pc.close()
            await empty_async_queue(self.rgb_queue)
            await self.signaling.close()

    async def clear_queues(self) -> None:
        for queue_name in ASYNC_QUEUE_NAMES:
            await empty_async_queue(getattr(self, f"{queue_name}_queue"))
        for queue_name in QUEUE_NAMES:
            empty_queue(getattr(self, f"{queue_name}_queue"))

    # TODO: May have to find some way to avoid hard coding the track order
    async def __handle_stream_tracks(self, track: VideoStreamTrack) -> None:
        if self.track_counter == 0:
            local_track: VideoStreamTrack = RGBProcessor(track)
            target_queue: asyncio.Queue = self.rgb_queue
        elif self.track_counter == 1:
            local_track: VideoStreamTrack = DepthProcessor(track)
            target_queue: asyncio.Queue = self.depth_queue
        elif self.track_counter == 2:
            local_track: VideoStreamTrack = SemanticProcessor(track)
            target_queue: asyncio.Queue = self.semantic_queue
        self.__set_async_components(local_track, target_queue)
        self.media_processor.addTrack(local_track)
        await self.media_processor.start()

        self.track_counter = (self.track_counter + 1) % 3

    def __setup_track_callbacks(self) -> None:
        self.media_processor = MediaBlackhole()

        @self.pc.on("track")
        async def on_track(track: VideoStreamTrack):
            if track.kind == "video":
                await self.__handle_stream_tracks(track)

    def __setup_datachannel_callbacks(self) -> None:
        @self.pc.on("datachannel")
        async def on_datachannel(channel: RTCDataChannel) -> None:
            self.data_channel = channel

            @self.data_channel.on("open")
            async def on_open() -> None:
                print("Data channel opened")

            @self.data_channel.on("message")
            async def on_message(message: bytes) -> None:
                print("Receiving data channel message")
                step_data: Dict[str, Any] = json.loads(message)
                if not step_data['reset']:
                    step_data['rgb'] = decompress_image(step_data['rgb'])
                    step_data['rgb'] = cv2.cvtColor(step_data['rgb'], cv2.COLOR_RGB2BGR)
                    step_data['depth'] = decompress_depth(step_data['depth'])
                    step_data['semantic'] = decompress_semantic(step_data['semantic'])

                    for key, val in step_data.items():
                        if key not in ['rgb', 'depth', 'semantic'] and isinstance(val, list):
                            step_data[key] = np.array(val)
                await self.loop.run_in_executor(None, self.step_queue.put, step_data.copy())

            @self.data_channel.on("close")
            def on_close() -> None:
                logging.info("Data channel closed")
                self.done.set()

            await self.send_action()

    def __set_async_components(
        self,
        component: BaseAsyncComponent,
        queue: Queue,
    ) -> None:
        component.set_loop(self.loop)
        component.set_input_queue(queue)

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def set_queue(self, queue_name: str, queue: Queue) -> None:
        setattr(self, f"{queue_name}_queue", queue)
