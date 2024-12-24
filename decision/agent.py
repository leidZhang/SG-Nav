import io
from typing import List, Dict, Any
from types import SimpleNamespace
from queue import Queue

import numpy as np


def dict_to_namespace(d):
    # Recursively convert dicts to SimpleNamespace
    return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})


class HabitatActuator:
    '''
    HabitatActuatorActuator class provides observation and execute the action.
    Cooperate with ProviderPeer to send the observation and receive the action.
    Will be deployed in the computer with the habitat simulator.
    '''

    CHANNELS: List[str] = ["rgb", "depth", "semantic"]

    def __init__(self) -> None:
        self.step: int = 0
        self.rgb_queue: Queue = None
        self.depth_queue: Queue = None
        self.state_queue: Queue = None
        self.action_queue: Queue = None
        self.first_reset: bool = True

    def reset(self) -> None:
        self.step = 0 # Reset the step counter
        state: dict = {"reset": True, 'step': -1} # Reset signal
        print(f"Sending state: {state}")
        self.state_queue.put(state.copy())

    def act(self, observations: dict) -> Dict[str, Any]:
        print("Sending observations to the server...")
        self.__transmit_observation(observations)

        action: Dict[str, Any] = self.__receive_action()
        print(f"Got action: {action}")
        print("====================================")
        self.step += 1
        return action

    def __transmit_observation(self, observations: dict) -> None:
        # Convert the Observations object to a dictionary to avoid pickling issues
        state: dict = {key: observations[key].tolist() for key in observations.keys() if key not in self.CHANNELS}
        state["reset"], state["step"] = False, self.step # Tell the remote peer that it is not a reset state
        for i, channel in enumerate(self.CHANNELS):
            state[channel] = observations[channel].copy()
            # getattr(self, f"{channel}_queue").put(observations[channel].copy())

        # print(f"Sending state: {state}")
        self.state_queue.put(state.copy())

    def __receive_action(self) -> Dict[str, Any]:
        return self.action_queue.get()

    def set_queue(self, queue_name: str, queue: Queue) -> None:
        setattr(self, f"{queue_name}_queue", queue)


class CognitiveUnit: # Not inheriting from Agent on purpose
    '''
    CognitiveUnit class receives observation and generate the action.
    Cooperate with ReceiverPeer to receive the observation and send the action.
    Will be deployed in the workstation.
    '''

    def __init__(self, core_agent) -> None:
        self.core_agent = core_agent
        self.observations_queue: Queue = None
        self.action_queue: Queue = None

    def reset(self) -> None:
        self.core_agent.reset()

    def act(self, observations) -> None:
        # observations: Dict[str, Any] = self.observations_queue.get()
        action: Dict[str, Any] = self.core_agent.act(observations)
        self.action_queue.put(action)

    def set_queue(self, queue_name: str, queue: Queue) -> None:
        setattr(self, f"{queue_name}_queue", queue)
