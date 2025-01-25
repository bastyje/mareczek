import os
import re
from typing import Optional

MODEL_DIR = 'models'
MODEL_PTH = 'model.pth'
LOG_FILE = 'training.log'
CONFIG_FILE = 'config.yaml'

DIR_PATTERN = re.compile(r'(.+)-(cnn|dense)-(ram|image)-(\d{4})')

class FileParams:

    def __init__(self, env_name: str, cnn: bool, obs_type: str):
        self.env_name = env_name.replace('/', '_')
        self.cnn = cnn
        self.obs_type = obs_type

    def get_model_dir(self, version: int) -> str:
        return os.path.join(
            MODEL_DIR,
            f'{self.env_name}-{"cnn" if self.cnn else "dense"}-{"ram" if self.obs_type else "image"}-{version:04d}')

    @staticmethod
    def _filter_dir(file: str, env_name: str, cnn: bool, obs_type: str) -> list[str]:
        match = DIR_PATTERN.match(file)
        return match \
            and match.group(1) == env_name \
            and match.group(2) == ('cnn' if cnn else 'dense') \
            and match.group(3) == obs_type

    def get_last_version(self) -> Optional[int]:
        versions = [int(f[-4:]) for f in os.listdir(MODEL_DIR) if self._filter_dir(f, self.env_name, self.cnn, self.obs_type)]
        return max(versions) if versions else None

    def get_last_model_path(self) -> Optional[str]:
        last_version = self.get_last_version()
        return self.get_model_path(last_version) if last_version is not None else None

    def get_model_path(self, version: int) -> str:
        return os.path.join(self.get_model_dir(version), MODEL_PTH)

    def get_log_path(self, version: int) -> str:
        return os.path.join(self.get_model_dir(version), LOG_FILE)

    def get_config_path(self, version: int) -> str:
        return os.path.join(self.get_model_dir(version), CONFIG_FILE)

    def get_new_version(self) -> int:
        last_version = self.get_last_version()
        return last_version + 1 if last_version is not None else 0

    def get_new_model_path(self) -> str:
        return self.get_model_path(self.get_new_version())