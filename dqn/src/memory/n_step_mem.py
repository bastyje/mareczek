from collections import deque
from typing import Iterator

from memory.transition import Transition


class NStepMemory:
    def __init__(self, n_steps: int, gamma: float):
        self.__n_steps = n_steps
        self.__memory = deque([], maxlen=n_steps)
        self.__gamma = gamma

    def store(self, transition: Transition) -> None:
        self.__memory.append(transition)

    def any(self) -> bool:
        return len(self.__memory) > 0

    def filled(self) -> bool:
        return len(self.__memory) == self.__n_steps

    def pop(self) -> Transition:
        state, action, _, _, _ = self.__memory[0]
        _, _, _, next_state, done = self.__memory[-1]
        reward = sum([self.__gamma ** i * r for i, (_, _, r, _, _) in enumerate(self.__memory)])
        self.__memory.popleft()
        return Transition(state, action, reward, next_state, done)

    def flush(self) -> Iterator[Transition]:
        while self.any():
            yield self.pop()