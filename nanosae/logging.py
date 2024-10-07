from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

try:
    import wandb
except ImportError:
    pass


@dataclass
class WandbConfig:
    project: str
    entity: str
    group: str
    name: str
    mode: str = "online"


class Logger(ABC):
    @abstractmethod
    def log(self, log_dict: dict[str, Any], step: int):
        pass


class WandbLogger(Logger):
    def __init__(self, config: WandbConfig):
        # Finish any existing runs
        wandb.finish()

        wandb.init(
            project=config.project,
            entity=config.entity,
            group=config.group,
            name=config.name,
            mode=config.mode,
        )
        self.wandb = wandb

    def __del__(self):
        wandb.finish()

    def set_config(self, config: dict[str, Any]):
        self.wandb.config.update(config)

    def log(self, log_dict: dict[str, Any], step: int):
        self.wandb.log(log_dict, step=step)
