from queue import Queue
from typing import Dict, Any
from types import SimpleNamespace

import numpy as np

from .sg_nav import CLIP_LLM_FMMAgent_NonPano


def dict_to_namespace(d):
    # Recursively convert dicts to SimpleNamespace
    return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})


class CognitiveUnit: # Not inheriting from Agent on purpose
    '''
    CognitiveUnit class receives observation and generate the action.
    Cooperate with ReceiverPeer to receive the observation and send the action.
    Will be deployed in the workstation.
    '''

    def __init__(self, core_agent: CLIP_LLM_FMMAgent_NonPano) -> None:
        self.core_agent = core_agent
        self.observations_queue: Queue = None
        self.action_queue: Queue = None

    def reset(self) -> None:
        print("Resetting core agent...")
        self.core_agent.reset()

    def act(self, observations: dict) -> None:
        # observations: Dict[str, Any] = self.observations_queue.get()
        for key, val in observations.items():
            if not isinstance(val, np.ndarray):
                observations[key] = np.array(val)

        action: Dict[str, Any] = self.core_agent.act(observations)
        self.action_queue.put(action)

    def set_queue(self, queue_name: str, queue: Queue) -> None:
        setattr(self, f"{queue_name}_queue", queue)