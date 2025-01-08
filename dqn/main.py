import os

import gymnasium as gym
import ale_py

import dqn
import ui
from train import train

args = ui.get_args()

gym.register_envs(ale_py)
env = gym.make(args.env, obs_type='grayscale')

network = dqn.create(args.cnn, env.observation_space.shape, env.action_space.n)
target_network = dqn.create(args.cnn, env.observation_space.shape, env.action_space.n)

model_dir = f'models{"-cnn" if args.cnn else ""}/' + args.env.replace('/', '_')
episodes = 1000

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

train(model_dir, episodes, env, network, target_network)
