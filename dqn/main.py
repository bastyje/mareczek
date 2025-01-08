import argparse

import gymnasium as gym
import ale_py

from dqn import DQN1
from train import train

envs = ['ALE/DonkeyKong-v5', 'ALE/SpaceInvaders-v5']
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='ALE/DonkeyKong-v5', choices=envs)
args = parser.parse_args()

gym.register_envs(ale_py)

if args.env == 'ALE/DonkeyKong-v5':
    env = gym.make('ALE/DonkeyKong-v5', obs_type='grayscale')

    network = DQN1()
    target_network = DQN1()
    model_dir = 'donkey-kong'
    episodes = 1000

    train(model_dir, episodes, env, network, target_network)
