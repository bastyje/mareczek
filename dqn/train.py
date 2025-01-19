import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

import logger
from dqn_agent import DQNAgent


__version_length = 3
def next_model_path(model_dir: str) -> str:
    previous = newest_model_path(model_dir)
    if previous is None:
        return os.path.join(model_dir, '0'.zfill(__version_length))
    previous_version = int(os.path.split(previous)[-1])
    return os.path.join(model_dir, str(previous_version + 1).zfill(__version_length))

def newest_model_path(model_dir: str) -> Optional[str]:
    if not os.path.exists(model_dir):
        return None
    versions = [int(f) for f in os.listdir(model_dir) if f.isdigit()]
    if not versions:
        return None
    return os.path.join(model_dir, str(max(versions)).zfill(__version_length))


def load_model(agent: DQNAgent, model_name: str) -> None:
    newest_model = newest_model_path(model_name)
    if newest_model is not None:
        agent.load_model(newest_model)
        logger.log_model_loaded(newest_model)


def save_model(agent: DQNAgent, model_name: str) -> None:
    new_model_version = next_model_path(model_name)
    agent.save_model(new_model_version)
    logger.log_model_saved(new_model_version)


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
            agent.try_update_target_network(episode)

            actions += 1

        scores.append(score.item())
        logger.log_episode_summary(episode, float(np.mean(scores)), agent.get_epsilon(), score.item())
        agent.update_epsilon()


def train(agent: DQNAgent, model_name: str, episodes: int, env: gym.Env) -> None:
    dtype = torch.float32
    scores = []

    load_model(agent, model_name)

    try:
        iterate(episodes, env, agent, dtype, scores)
        logger.log_training_done()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        save_model(agent, model_name)