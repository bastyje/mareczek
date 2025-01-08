import torch
import gymnasium as gym
from torch import nn

from replay_mem import ReplayMemory, Transition


class DQNAgent:
    device: torch.device
    __env: gym.Env
    __network: nn.Module
    __target_network: nn.Module
    __memory: ReplayMemory
    __optimizer: torch.optim.Optimizer

    __EPSILON: float = 1.0
    __EPSILON_DECAY: float = 0.995
    __EPSILON_MIN: float = 0.01
    __GAMMA: float = 0.99
    __BATCH_SIZE: int = 32
    __LEARNING_RATE: float = 0.001
    __TARGET_UPDATE: int = 10

    def __init__(self, env: gym.Env, network: nn.Module, target_network: nn.Module):
        self.device = self.__get_device()
        self.__env = env
        self.__network = network.to(self.device)
        self.__target_network = target_network.to(self.device)
        self.__target_network.load_state_dict(self.__network.state_dict())
        self.__target_network.eval()
        self.__memory = ReplayMemory(10000)
        self.__optimizer = torch.optim.Adam(self.__network.parameters(), lr=self.__LEARNING_RATE)

    def act_best(self, state) -> int:
        self.__network.eval()
        with torch.no_grad():
            max_action = torch.argmax(self.__network(torch.tensor(state, device=self.device, dtype=torch.float32)), dim=1)
        self.__network.train()
        return max_action.item()

    def act(self, state: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() <= self.__EPSILON:
            return torch.tensor([self.__env.action_space.sample()], device=self.device, dtype=torch.int64)
        self.__network.eval()
        with torch.no_grad():
            max_action = torch.argmax(self.__network(state), dim=1)
        self.__network.train()
        return max_action

    def step(self, state, action, reward, next_state, done):
        self.__memory.push(state, action, reward, next_state, done)
        if len(self.__memory) > self.__BATCH_SIZE:
            self.update_model()

    def update_model(self):
        states, actions, rewards, next_states, dones = zip(*self.__memory.sample(self.__BATCH_SIZE))

        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)

        q_values = self.__network(states).gather(1, actions)

        target_q_values = torch.zeros(self.__BATCH_SIZE, device=self.device)
        target_q_values[~torch.tensor(dones)] = self.evaluate(self.__target_network, next_states[~torch.tensor(dones)])
        target_q_values = rewards + self.__GAMMA * target_q_values.unsqueeze(1)

        loss = torch.nn.functional.mse_loss(q_values, target_q_values)

        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()

    def try_update_target_network(self, episodes: int):
        if episodes % self.__TARGET_UPDATE == 0:
            self.__target_network.load_state_dict(self.__network.state_dict())

    def update_epsilon(self):
        self.__EPSILON = max(self.__EPSILON_MIN, self.__EPSILON * self.__EPSILON_DECAY)

    def get_epsilon(self):
        return self.__EPSILON

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
    def __get_device():
        return torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )