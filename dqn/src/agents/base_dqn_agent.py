from abc import ABC, abstractmethod

import gymnasium as gym
import torch
from torch import nn

from memory.replay_mem import ReplayMemory


class BaseDQNAgent(ABC):

    device: torch.device
    _env: gym.Env
    _network: nn.Module
    _target_network: nn.Module
    _memory: ReplayMemory
    _optimizer: torch.optim.Optimizer

    _epsilon: float
    _epsilon_decay: float
    _epsilon_min: float
    _gamma: float
    _batch_size: int
    _learning_rate: float
    _target_update: int

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
        self._init_parameters(epsilon, epsilon_decay, epsilon_min, gamma, batch_size, learning_rate, target_update)
        self.device = self._get_device()
        self._env = env
        self._network = network.to(self.device)
        self._target_network = target_network.to(self.device)
        self._target_network.load_state_dict(self._network.state_dict())
        self._target_network.eval()
        self._memory = ReplayMemory(buffer_size)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=self._learning_rate)

    def _init_parameters(
            self,
            epsilon: float,
            epsilon_decay: float,
            epsilon_min: float,
            gamma: float,
            batch_size: int,
            learning_rate: float,
            target_update: int):
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._gamma = gamma
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._target_update = target_update

    def update_model(self):
        states, actions, rewards, next_states, dones = self._memory.sample(self._batch_size)
        q_values = self._network(states).gather(1, actions)

        target_q_values = self._target_network(next_states).max(1).values
        target_q_values[dones] = 0
        target_q_values = rewards + self._gamma * target_q_values.unsqueeze(1)

        loss = torch.nn.functional.mse_loss(q_values, target_q_values)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    @abstractmethod
    def step(self, step, action, reward, next_state, done) -> None:
        raise NotImplementedError()

    def act(self, state: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() <= self._epsilon:
            return torch.tensor([self._env.action_space.sample()], device=self.device, dtype=torch.int64)
        self._network.eval()
        with torch.no_grad():
            max_action = torch.argmax(self._network(state), dim=1)
        self._network.train()
        return max_action

    def try_update_target_network(self, episodes: int):
        if episodes % self._target_update == 0:
            self._target_network.load_state_dict(self._network.state_dict())

    def update_epsilon(self):
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

    def get_epsilon(self) -> float:
        return self._epsilon

    def save_model(self, path: str):
        torch.save(self._network.state_dict(), path)

    def load_model(self, path: str):
        self._network.load_state_dict(torch.load(path))
        self._target_network.load_state_dict(self._network.state_dict())

    @staticmethod
    def evaluate(network: nn.Module, state: torch.Tensor) -> torch.Tensor:
        network.eval()
        with torch.no_grad():
            values = network(state)
        network.train()
        return values.max(1).values

    @staticmethod
    def _get_device() -> torch.device:
        return torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )