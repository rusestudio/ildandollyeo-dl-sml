from hydra import compose, initialize
from omegaconf import DictConfig

def load_config(overrides=()) -> DictConfig:
    with initialize(version_base="1.3", config_path="config"):
        return compose(config_name="config", overrides=list(overrides))