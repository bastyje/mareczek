import logging
import os
import sys
from datetime import datetime


logger = logging.getLogger('dqn')
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def init(model_name: str):
    file_path = os.path.join('logs', f'{model_name}_{datetime.now().strftime("%Y%m%d%H%M%S")}.log')
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def log_device(device: str):
    logger.info(f'Device: {device}')


def log_episode_summary(episode: int, score: float, epsilon: float, total_score: float):
    logger.info(f'Finished episode {episode}; total score {total_score:.0f}; avg score {score:.0f}; epsilon {epsilon:.10f}')


def log_model_loaded(model_file: str):
    logger.info(f'Model loaded: {model_file}')


def log_model_saved(model_file: str):
    logger.info(f'Model saved: {model_file}')


def log_training_done():
    logger.info('Training done')