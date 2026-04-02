import os
import toml
from typing import Any, Optional

class Config:
    _instance = None
    _config_data = None
    _config_path = None

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._config_path = config_path or os.path.join(os.getcwd(), "config.toml")
        return cls._instance

    def _load_config(self):
        if self._config_data is not None:
            return

        if not os.path.exists(self._config_path):
            raise FileNotFoundError(f"config file not found: {self._config_path}")

        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._config_data = toml.load(f)
        except toml.TomlDecodeError as e:
            raise ValueError(f"config file format error ({self._config_path}): {e}")
        except Exception as e:
            raise RuntimeError(f"failed to read config file ({self._config_path}): {e}")

    def get(self, key: str, default: Any = None) -> Any:
        self._load_config()

        keys = key.split('.')
        value = self._config_data

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"config item '{key}' not found")

    def reload(self):
        self._config_data = None
        self._load_config()

config = Config()