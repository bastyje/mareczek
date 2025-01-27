import ale_py
import gymnasium as gym
import torch

from agents.n_step_dqn_agent import NStepDQNAgent
from dqn import DQN1 as DQN

gym.register_envs(ale_py)
environment = gym.make("ALE/DonkeyKong-v5", obs_type='grayscale')
network = DQN(environment.observation_space.shape, environment.action_space.n)
target_network = DQN(environment.observation_space.shape, environment.action_space.n)

agent = NStepDQNAgent(environment, network, target_network, n_steps=3)
state = torch.tensor(environment.reset()[0], device=agent.device, dtype=torch.float32)
action = agent.act(state)
next_state, reward, done, _, _ = environment.step(action)
agent.step(state, action, reward, next_state, done)
