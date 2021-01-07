from typing import Optional

from torch.nn import Module
from torch.utils.data import DataLoader

import torch

from src import settings

from src.utils import logger

from src.trainer import TransferLearningTrainer

from src.attack import LinfPGDAttack

from src.cli.utils import get_train_dataset, get_test_dataset
from src.cli.utils import get_model


def tl(model, num_classes, dataset, k, teacher):
    """transform leanring"""
    save_name = f"tl_{model}_{dataset}_{k}_{teacher}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    trainer = TransferLearningTrainer(
        k=k,
        teacher_model_path=str(settings.model_dir / teacher),
        model=get_model(model, num_classes, k),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-n", "--num_classes", type=int)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-k", "--k", type=int)
    parser.add_argument("-t", "--teacher", type=str)

    args = parser.parse_args()

    tl(model=args.model, num_classes=args.num_classes, dataset=args.dataset, k=args.k, teacher=args.teacher)
