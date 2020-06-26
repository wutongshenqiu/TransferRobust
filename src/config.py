from pydantic import BaseSettings, validator

from typing import List


class Settings(BaseSettings):
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

    class Config:
        env_file = '.env'


settings = Settings(_env_file="config.env")

print(settings.dict())
