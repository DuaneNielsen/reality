from typing import Any, Optional, overload, Typing, Sequence, Iterable, Union, Callable
from enum import Enum
import madrona_escape_room

AGENT_ID_SIZE: int

SELF_OBSERVATION_SIZE: int

STEPS_REMAINING_SIZE: int

class SimManager:
    """
    None
    """

    def __init__(self, exec_mode: madrona_escape_room.madrona.ExecMode, gpu_id: int, num_worlds: int, rand_seed: int, auto_reset: bool, enable_batch_renderer: bool = False) -> None:
        ...
    
    def action_tensor(self) -> madrona_escape_room.madrona.Tensor:
        ...
    
    def depth_tensor(self) -> madrona_escape_room.madrona.Tensor:
        ...
    
    def done_tensor(self) -> madrona_escape_room.madrona.Tensor:
        ...
    
    def reset_tensor(self) -> madrona_escape_room.madrona.Tensor:
        ...
    
    def reward_tensor(self) -> madrona_escape_room.madrona.Tensor:
        ...
    
    def rgb_tensor(self) -> madrona_escape_room.madrona.Tensor:
        ...
    
    def self_observation_tensor(self) -> madrona_escape_room.madrona.Tensor:
        ...
    
    def step(self) -> None:
        ...
    
    def steps_remaining_tensor(self) -> madrona_escape_room.madrona.Tensor:
        ...
    
TOTAL_OBSERVATION_SIZE: int

