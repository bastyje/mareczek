import gymnasium as gym
import ale_py

import dqn
import ui
from dqn_agent import DQNAgent
from train import newest_model_path

args = ui.get_args()

gym.register_envs(ale_py)
env = gym.make(args.env, obs_type='grayscale', render_mode='human' if args.render else 'rgb_array')

network = dqn.create(args.cnn, env.observation_space.shape, env.action_space.n)
target_network = dqn.create(args.cnn, env.observation_space.shape, env.action_space.n)

terminated = False
state, _ = env.reset()
total_reward = 0

agent = DQNAgent(env, network, target_network)
model_dir = f'models{"-cnn" if args.cnn else ""}/' + args.env.replace('/', '_')
newest_model = newest_model_path(model_dir)
agent.load_model(newest_model)

while not terminated:
    action = agent.act(state)
    state, reward, terminated, truncated, _ = env.step(action)
    if args.render:
        env.render()
    total_reward += reward
    terminated = terminated or truncated
print(f'Total reward: {total_reward}')
env.close()