import random

from collections import deque
from typing import Tuple

import torch

from memory.transition import Transition


class ReplayMemory:
    __memory: deque[Transition]

    def __init__(self, capacity: int):
        self.__memory = deque([], maxlen=capacity)

    def __iter__(self):
        return iter(self.__memory)

    def push(self, *args: Transition) -> None:
        self.__memory.append(Transition(*args))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[bool]]:
        states, actions, rewards, next_states, dones = zip(*random.sample(self.__memory, batch_size))
        return torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(next_states), dones

    def __len__(self) -> int:
        return len(self.__memory)