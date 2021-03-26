from typing import Optional

from torch.nn import Module
from torch.utils.data import DataLoader

import torch

from src import settings

from src.utils import logger

from src.trainer import ParsevalTransferLearningTrainer

from src.cli.utils import get_train_dataset, get_test_dataset
from src.cli.utils import get_model


def ptl(model, num_classes, dataset, beta, k, teacher, freeze_bn):
    """transform leanring"""

    save_name = f"ptl_{freeze_bn}_{model}_{dataset}_{beta}_{k}_from_{teacher}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")

    trainer = ParsevalTransferLearningTrainer(
        beta=beta,
        k=k,
        teacher_model_path=str(settings.model_dir / teacher),\
        model=get_model(model=model, num_classes=num_classes, k=k),
        train_loader=get_train_dataset(dataset=dataset),
        test_loader=get_test_dataset(dataset=dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth",
        freeze_bn=freeze_bn
    )

    trainer.train(f"{settings.model_dir / save_name}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-n", "--num_classes", type=int)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-k", "--k", type=int)
    parser.add_argument("-t", "--teacher", type=str)
    parser.add_argument("-b", "--beta", type=float)
    parser.add_argument("--freeze-bn", action="store_true")
    

    args = parser.parse_args()

    ptl(model=args.model, num_classes=args.num_classes, dataset=args.dataset, beta=args.beta, k=args.k, 
            teacher=args.teacher, freeze_bn=args.freeze_bn)
