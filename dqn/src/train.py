import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import yaml

from utils import logger
from agents.dqn_agent import DQNAgent
from utils.files import FileParams


def iterate(episodes: int, env: gym.Env, agent: DQNAgent, dtype: torch.dtype, scores: list) -> None:
    logger.log_device(str(agent.device))
    for episode in range(episodes):
        state = torch.tensor(env.reset()[0], device=agent.device, dtype=dtype)

        score = 0
        done = False
        actions = 0
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            reward = torch.tensor([reward], device=agent.device, dtype=dtype)
            next_state = torch.tensor(next_state, device=agent.device, dtype=dtype)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            score += reward

            state = next_state

            actions += 1

        agent.try_update_target_network(episode)
        scores.append(score.item())
        logger.log_episode_summary(episode, float(np.mean(scores)), agent.get_epsilon(), score.item())
        agent.update_epsilon()


def train(agent: DQNAgent, params: FileParams, episodes: int, env: gym.Env, previous_version: Optional[int]) -> None:
    dtype = torch.float32
    scores = []

    if previous_version is not None:
        agent.load_model(params.get_model_path(previous_version))
    new_model_version = params.get_new_version()
    os.makedirs(params.get_model_dir(new_model_version), exist_ok=True)
    with open(params.get_config_path(new_model_version), 'w') as f:
        f.write(yaml.dump(agent.get_hyperparameters()))
    logger.init(params.get_log_path(new_model_version))
    logger.log_hyperparameters(agent.get_hyperparameters())

    try:
        iterate(episodes, env, agent, dtype, scores)
        logger.log_training_done()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        agent.save_model(params.get_model_path(new_model_version))