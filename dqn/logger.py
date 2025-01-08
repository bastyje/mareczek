import logging


logger = logging.getLogger('dqn')
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def log_episode(episode: int):
    logger.info(f'Episode {episode}')


def try_log_action(actions: int, score: float, epsilon: float):
    if actions % 100 == 0:
        logger.info(f'Actions: {actions}, Epsilon: {epsilon:.10f}, Score: {score:.0f}')


def log_model_loaded(model_file: str):
    logger.info(f'Model loaded: {model_file}')


def log_model_saved(model_file: str):
    logger.info(f'Model saved: {model_file}')


def log_training_done():
    logger.info('Training done')