import torch
import gymnasium as gym
from torch import nn

from replay_mem import ReplayMemory


class DQNAgent:
    device: torch.device
    __env: gym.Env
    __network: nn.Module
    __target_network: nn.Module
    __memory: ReplayMemory
    __optimizer: torch.optim.Optimizer

    __epsilon: float
    __epsilon_decay: float
    __epsilon_min: float
    __gamma: float
    __batch_size: int
    __learning_rate: float
    __target_update: int

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
        self.__init_parameters(epsilon, epsilon_decay, epsilon_min, gamma, batch_size, learning_rate, target_update)
        self.device = self.__get_device()
        self.__env = env
        self.__network = network.to(self.device)
        self.__target_network = target_network.to(self.device)
        self.__target_network.load_state_dict(self.__network.state_dict())
        self.__target_network.eval()
        self.__memory = ReplayMemory(buffer_size)
        self.__optimizer = torch.optim.Adam(self.__network.parameters(), lr=self.__learning_rate)

    def __init_parameters(
            self,
            epsilon: float,
            epsilon_decay: float,
            epsilon_min: float,
            gamma: float,
            batch_size: int,
            learning_rate: float,
            target_update: int):
        self.__epsilon = epsilon
        self.__epsilon_decay = epsilon_decay
        self.__epsilon_min = epsilon_min
        self.__gamma = gamma
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__target_update = target_update

    def act_best(self, state) -> int:
        self.__network.eval()
        with torch.no_grad():
            max_action = torch.argmax(self.__network(torch.tensor(state, device=self.device, dtype=torch.float32)), dim=1)
        self.__network.train()
        return max_action.item()

    def act(self, state: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() <= self.__epsilon:
            return torch.tensor([self.__env.action_space.sample()], device=self.device, dtype=torch.int64)
        self.__network.eval()
        with torch.no_grad():
            max_action = torch.argmax(self.__network(state), dim=1)
        self.__network.train()
        return max_action

    def step(self, state, action, reward, next_state, done):
        self.__memory.push(state, action, reward, next_state, done)
        if len(self.__memory) > self.__batch_size:
            self.update_model()

    def update_model(self):
        states, actions, rewards, next_states, dones = self.__memory.sample(self.__batch_size)
        q_values = self.__network(states).gather(1, actions)

        target_q_values = self.__target_network(next_states).max(1).values
        target_q_values[dones] = 0
        target_q_values = rewards + self.__gamma * target_q_values.unsqueeze(1)

        # target_q_values = torch.zeros(self.__batch_size, device=self.device)
        # target_q_values[~torch.tensor(dones)] = self.evaluate(self.__target_network, next_states[~torch.tensor(dones)])
        # target_q_values = rewards + self.__gamma * target_q_values.unsqueeze(1)

        loss = torch.nn.functional.mse_loss(q_values, target_q_values)
        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()

    def try_update_target_network(self, episodes: int):
        if episodes % self.__target_update == 0:
            self.__target_network.load_state_dict(self.__network.state_dict())

    def update_epsilon(self):
        self.__epsilon = max(self.__epsilon_min, self.__epsilon * self.__epsilon_decay)

    def get_epsilon(self) -> float:
        return self.__epsilon

    def save_model(self, path: str):
        torch.save(self.__network.state_dict(), path)

    def load_model(self, path: str):
        self.__network.load_state_dict(torch.load(path))
        self.__target_network.load_state_dict(self.__network.state_dict())

    @staticmethod
    def evaluate(network: nn.Module, state: torch.Tensor) -> torch.Tensor:
        network.eval()
        with torch.no_grad():
            values = network(state)
        network.train()
        return values.max(1).values

    @staticmethod
    def __get_device() -> torch.device:
        return torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )