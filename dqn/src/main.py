import gymnasium as gym
import ale_py
import yaml

import dqn
import reward_modifier
from utils import ui
from agents.dqn_agent import DQNAgent
from agents.n_step_dqn_agent import NStepDQNAgent
from train import train
from utils.files import FileParams

args = ui.get_args()

gym.register_envs(ale_py)
env = gym.make(args.env, obs_type=args.obs_type)

is_ram = args.obs_type == 'ram'
if is_ram:
    input_size = env.observation_space.shape[0]
else:
    input_size = env.observation_space.shape[0] * env.observation_space.shape[1]

hidden_size = 64
output_size = env.action_space.n

network = dqn.DQN1(input_size, hidden_size, output_size, is_ram) if not args.cnn else dqn.DQN2(env.observation_space.shape, output_size)
target_network = dqn.DQN1(input_size, hidden_size, output_size, is_ram) if not args.cnn else dqn.DQN2(env.observation_space.shape, output_size)

episodes = 1000

with open('agent-config.yaml', 'r') as f:
    agent_config = yaml.safe_load(f)

modifier = None
if args.env.startswith('ALE/SpaceInvaders'):
    modifier = reward_modifier.space_invaders

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
        buffer_size=agent_config['buffer_size'],
        # reward_modifier=modifier
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
        n_steps=args.steps,
        # reward_modifier=modifier
    )

file_params = FileParams(args.env, args.cnn, args.obs_type)
train(agent, file_params, episodes, env, args.continue_from)
