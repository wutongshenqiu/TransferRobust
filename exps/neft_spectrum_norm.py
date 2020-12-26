from typing import Optional

from torch.nn import Module
from torch.utils.data import DataLoader

import torch

from src import settings

from src.utils import logger

from src.trainer import SpectralNormTransferLearningTrainer

from src.attack import LinfPGDAttack

from src.networks import (resnet18, wrn34_10, make_blocks,
                          parseval_retrain_wrn34_10, parseval_resnet18,
                          SupportedAllModuleType)

from src.utils import (get_cifar_test_dataloader, get_cifar_train_dataloader,
                       get_mnist_test_dataloader, get_mnist_train_dataloader,
                       get_svhn_test_dataloader, get_svhn_train_dataloder)


def sn_tl(model, num_classes, dataset, k, teacher, power_iter, norm_beta):
    """transform leanring"""

    save_name = f"sntl_{power_iter}_{norm_beta}_{model}_{dataset}_{k}_{teacher}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")

    trainer = SpectralNormTransferLearningTrainer(
        k=k,
        teacher_model_path=str(settings.model_dir / teacher),\
        model=parseval_retrain_wrn34_10(k=k, num_classes=num_classes),
        train_loader=get_cifar_train_dataloader(dataset=dataset),
        test_loader=get_cifar_test_dataloader(dataset=dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth",
        power_iter=power_iter,
        norm_beta=norm_beta
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
    parser.add_argument("--power-iter", type=int, default=1)
    parser.add_argument("--norm-beta", type=float, default=1.0)

    args = parser.parse_args()

    sn_tl(model=args.model, num_classes=args.num_classes, dataset=args.dataset, k=args.k, teacher=args.teacher, 
            power_iter=args.power_iter, norm_beta=args.norm_beta)
