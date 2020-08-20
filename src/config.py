from pydantic import BaseSettings, validator

from typing import List, Optional, Union
from pathlib import PurePath


ENV_PATH = PurePath(__file__).parent / "config.env"


class Settings(BaseSettings):
    root_dir: PurePath = PurePath(__file__).parent.parent
    log_dir: PurePath = root_dir / "logs"
    source_dir: PurePath = root_dir / "src"
    logger_config_file: PurePath = source_dir / "logger_config.toml"
    tensorboard_log_dir: PurePath = root_dir / "runs"

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
    log_file: Union[str, PurePath] = log_dir / "cifar100_robust_plus_regularization_blocks8_lambda_0.05.log"
    # log_file: Union[str, PurePath] = log_dir / "tl_cifar100_robust_plus_regularization_attack.log"
    # log_file: Union[str, PurePath] = log_dir / "parseval_tl_cifar100_robust_plus_regularization_lambda1_beta3e-4_attack.log"
    # log_file: Union[str, PurePath] = log_dir / "parseval_tl_cifar100_robust_plus_regularization_lambda1_beta6e-4_attack.log"
    # log_file: Union[str, PurePath] = log_dir / "parseval_tl_cifar100_robust_plus_regularization_lambda1_beta1e-3_attack.log"
    # log_file: Union[str, PurePath] = log_dir / "parseval_tl_cifar100_robust_plus_regularization_blocks4_lambda1_beta6e-4.log"
    # log_file: Union[str, PurePath] = log_dir / "parseval_tl_cifar100_robust_plus_regularization_blocks4_lambda1_beta3e-4.log"
    # log_file: Union[str, PurePath] = log_dir / "parseval_tl_cifar100_robust_plus_regularization_blocks4_lambda1_beta1e-3.log"
    # log_file: Union[str, PurePath] = log_dir / "parseval_tl_cifar100_pgd7_blocks6_lambda1_beta3e-4.log"
    # log_file: Union[str, PurePath] = log_dir / "parseval_tl_cifar100_pgd7_blocks8_lambda1_beta6e-4.log"
    # log_file: Union[str, PurePath] = log_dir / "parseval_tl_cifar100_pgd7_blocks8_lambda1_beta1e-3.log"
    # log_file: Union[str, PurePath] = log_dir / "parseval_tl_cifar100_pgd7_lambda1_beta3e-4_attack.log"
    # log_file: Union[str, PurePath] = log_dir / "parseval_tl_cifar100_pgd7_lambda1_beta6e-4_attack.log"
    # log_file: Union[str, PurePath] = log_dir / "parseval_tl_cifar100_pgd7_lambda1_beta1e-3_attack.log"
    # log_file: Union[str, PurePath] = log_dir / "tl_cifar100_robust_plus_regularization_blocks8_lambda1.log"

    @validator("logger_name")
    def check_logger_name(cls, v):
        if v not in {"StreamLogger", "FileLogger"}:
            raise ValueError("unsupported logger type!")
        return v

    class Config:
        env_file = '.env'


settings = Settings(_env_file=ENV_PATH)
