import random

from collections import namedtuple, deque
from typing import List, Tuple

import torch

Transition = namedtuple(
    typename='Transition',
    field_names=('state', 'action', 'reward', 'next_state', 'done'))

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