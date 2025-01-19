import gymnasium as gym
from torch import nn

from agents.base_dqn_agent import BaseDQNAgent


class DQNAgent(BaseDQNAgent):

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
            buffer_size: int = 10000):
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

    def step(self, state, action, reward, next_state, done):
        self._memory.push(state, action, reward, next_state, done)
        if len(self._memory) > self._batch_size:
            self.update_model()