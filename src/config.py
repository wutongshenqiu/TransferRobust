from pydantic import BaseSettings, validator

from typing import List, Optional, Union
from pathlib import PurePath


ENV_PATH = PurePath(__file__).parent / "config.env"


class Settings(BaseSettings):
    root_dir: PurePath = PurePath(__file__).parent.parent
    log_dir: PurePath = root_dir / "logs"
    source_dir: PurePath = root_dir / "src"
    logger_config_file: PurePath = source_dir / "logger_config.toml"

    device: str = "cuda: 0"

    momentum: float = 0.9
    weight_decay: float = 5e-4

    batch_size: int = 128
    num_worker: int = 4

    start_lr: float = 0.1
    train_epochs: int = 100
    warm_up_epochs: int = 1

    # CE for cross-entropy and MSE for mean-square-error
    criterion: str = "CE"

    @validator("criterion")
    def criterion_must_be_correct(cls, v):
        if v == "CE":
            return "CrossEntropyLoss"
        elif v == "MSE":
            return "MSELoss"
        else:
            raise ValueError(f"criterion `{v}` is not supported!")

    # drop lr to lr*decrease_rate when reach each milestone
    milestones: List[int] = [40, 70, 90]
    decrease_rate: float = 0.2

    # whether use multiple GPUs
    parallelism: bool = False

    dataset_name: str = "cifar10"

    @validator("dataset_name")
    def check_dataset_name(cls, v):
        if not v:
            raise ValueError("`dataset_name` must be specified")
        if v not in {"cifar10", "cifar100"}:
            raise ValueError("`dataset_name` must be specified as `cifar10` or `cifar100`")
        return v

    logger_name: str = "StreamHandler"
    log_file: Optional[Union[str, PurePath]] = log_dir / "attack.log"

    @validator("logger_name")
    def check_logger_name(cls, v):
        if v not in {"StreamLogger", "FileLogger"}:
            raise ValueError("unsupported logger type!")
        return v

    class Config:
        env_file = '.env'


settings = Settings(_env_file=ENV_PATH)

