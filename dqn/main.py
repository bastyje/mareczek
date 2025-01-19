import os

import gymnasium as gym
import ale_py
import yaml

import dqn
import dqn_agent
import ui
from train import train

args = ui.get_args()

gym.register_envs(ale_py)
env = gym.make(args.env, obs_type='grayscale')

input_size = env.observation_space.shape[0] * env.observation_space.shape[1]
hidden_size = 128
output_size = env.action_space.n

network = dqn.DQN1(input_size, hidden_size, output_size) if not args.cnn else dqn.DQN2(env.observation_space.shape, output_size)
target_network = dqn.DQN1(input_size, hidden_size, output_size) if not args.cnn else dqn.DQN2(env.observation_space.shape, output_size)

model_dir = f'models{"-cnn" if args.cnn else ""}/' + args.env.replace('/', '_')
episodes = 1000

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

with open('agent-config.yaml', 'r') as f:
    agent_config = yaml.safe_load(f)['parameters']

agent = dqn_agent.DQNAgent(
    env,
    network,
    target_network,
    epsilon=agent_config['epsilon'],
    epsilon_decay=agent_config['epsilon_decay'],
    epsilon_min=agent_config['epsilon_min'],
    gamma=agent_config['gamma'],
    batch_size=agent_config['batch_size'],
    learning_rate=agent_config['learning_rate'],
    target_update=agent_config['target_update'],
    buffer_size=agent_config['buffer_size']
)

train(agent, model_dir, episodes, env)
