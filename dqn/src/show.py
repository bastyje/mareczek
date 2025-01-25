import gymnasium as gym
import ale_py
import torch
import yaml

import dqn
from agents.n_step_dqn_agent import NStepDQNAgent
from utils import ui
from agents.dqn_agent import DQNAgent
from train import newest_model_path

args = ui.get_args()

gym.register_envs(ale_py)
env = gym.make(args.env, obs_type=args.obs_type, render_mode='human' if args.render else 'rgb_array')

if args.obs_type == 'ram':
    input_size = env.observation_space.shape[0]
else:
    input_size = env.observation_space.shape[0] * env.observation_space.shape[1]

hidden_size = 128
output_size = env.action_space.n

network = dqn.DQN1(input_size, hidden_size, output_size) if not args.cnn else dqn.DQN2(env.observation_space.shape, output_size)
target_network = dqn.DQN1(input_size, hidden_size, output_size) if not args.cnn else dqn.DQN2(env.observation_space.shape, output_size)

with open('agent-config.yaml', 'r') as f:
    agent_config = yaml.safe_load(f)['train-parameters']

if args.steps == 1:
    agent = DQNAgent(
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
else:
    agent = NStepDQNAgent(
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
        buffer_size=agent_config['buffer_size'],
        n_steps=args.steps
    )

model_dir = f'models{"-cnn" if args.cnn else ""}{"-ram" if args.obs_type == "ram" else ""}/' + args.env.replace('/', '_')
newest_model = newest_model_path(model_dir)
agent.load_model(newest_model)

terminated = False
state, _ = env.reset()
total_reward = 0

while not terminated:
    action = agent.act(torch.tensor(state, device=agent.device, dtype=torch.float32))
    state, reward, terminated, truncated, _ = env.step(action)
    if args.render:
        env.render()
    total_reward += reward
    terminated = terminated or truncated
print(f'Total reward: {total_reward}')
env.close()