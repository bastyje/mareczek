from torch import nn
import gymnasium as gym

from agents.base_dqn_agent import BaseDQNAgent
from memory.n_step_mem import NStepMemory


class NStepDQNAgent(BaseDQNAgent):

    def __init__(
            self,
            env: gym.Env,
            network: nn.Module,
            target_network: nn.Module,
            epsilon: float = 1.0,
            epsilon_decay: float = 0.995,
            epsilon_min: float = 0.01,
            gamma: float = 0.99,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            target_update: int = 10,
            buffer_size: int = 10000,
            n_steps: int = 3):
        super().__init__(
            env,
            network,
            target_network,
            epsilon,
            epsilon_decay,
            epsilon_min,
            gamma,
            batch_size,
            learning_rate,
            target_update,
            buffer_size)
        self.__n_step_buffer = NStepMemory(n_steps, self._gamma)

    def step(self, step, action, reward, next_state, done):
        self.__n_step_buffer.store((step, action, reward, next_state, done))
        if self.__n_step_buffer.filled():
            transition = self.__n_step_buffer.pop()
            self._memory.push(*transition)
        if done:
            for transition in self.__n_step_buffer.flush():
                self._memory.push(*transition)
        if len(self._memory) >= self._batch_size:
            self.update_model()


