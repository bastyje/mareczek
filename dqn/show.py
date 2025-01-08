import argparse

import gymnasium as gym
import ale_py

from dqn import DQN1
from dqn_agent import DQNAgent

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true')
args = parser.parse_args()

gym.register_envs(ale_py)
env = gym.make('ALE/DonkeyKong-v5', obs_type='grayscale', render_mode='human' if args.render else 'rgb_array')

terminated = False
state, _ = env.reset()
agent = DQNAgent(env, DQN1(), DQN1())
agent.load_model('dqn_agent.pth')
total_reward = 0
while not terminated:
    action = agent.act(state)
    state, reward, terminated, truncated, _ = env.step(action)
    if args.render:
        env.render()
    total_reward += reward
    terminated = terminated or truncated
print(f'Total reward: {total_reward}')
env.close()