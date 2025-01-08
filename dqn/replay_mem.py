import random

from collections import namedtuple, deque
from typing import List

Transition = namedtuple(
    typename='Transition',
    field_names=('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    __memory: deque[Transition]

    def __init__(self, capacity: int):
        self.__memory = deque([], maxlen=capacity)

    def push(self, *args: Transition) -> None:
        self.__memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.__memory, batch_size)

    def __len__(self) -> int:
        return len(self.__memory)