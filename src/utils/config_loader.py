import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, *keys, default=None):
        """Get nested config value"""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default

def load_config(config_path="config/config.yaml"):
    return Config(config_path)
