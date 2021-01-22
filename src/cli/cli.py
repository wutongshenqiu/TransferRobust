# TODO
# 避免 options 重复

import functools
from typing import Callable, Iterable, Reversible, Union, Iterator

import click

import torch

from .utils import (DefaultDataset, SupportDatasetList,
                    DefaultModel, SupportModelList,
                    SupportParsevalModelList, SupportNormalModelList,
                    get_test_dataset, get_train_dataset,
                    get_model)

from src import settings

from src.utils import logger

from src.trainer import (TransferLearningTrainer, LWFTransferLearningTrainer,
                         ParsevalTransferLearningTrainer, RetrainTrainer,
                         ParsevalRetrainTrainer, NormalTrainer,
                         ADVTrainer, RobustPlusSingularRegularizationTrainer,
                         BNTransferLearningTrainer)

from src.attack import LinfPGDAttack

_BasicOptions = [
    click.option("-m", "--model", type=click.Choice(SupportModelList),
                 default=DefaultModel, show_default=True, help="neural network"),
    click.option("-n", "--num_classes", type=int,
                 default=10, show_default=True, help="number of classes"),
    click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
                 default=DefaultDataset, show_default=True, help="dataset"),
]


@click.group()
def cli():
    ...


def apply_options(options: Union[Iterable, Reversible]):
    def _decorators(f: Callable):
        @functools.wraps(f)
        def _apply():
            nonlocal f
            for option in reversed(options):
                f = option(f)
            return f

        return _apply

    return _decorators


def composed(decs):
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f
    return deco


def apply_test(f):
    return composed(_BasicOptions)(f)


@cli.command()
# @apply_options(_BasicOptions)
@click.option("-m", "--model", type=click.Choice(SupportModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
@click.option("-k", "--k", type=int, required=True,
              help="trainable blocks from last")
@click.option("-t", "--teacher", type=str, required=True,
              help="filename of teacher model")
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


@cli.command()
@click.option("-m", "--model", type=click.Choice(SupportNormalModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
@click.option("-l", "--lambda_", type=float, required=True,
              help="penalization rate of feature representation between teacher and student")
@click.option("-t", "--teacher", type=str, required=True,
              help="filename of teacher model")
def lwf(model, num_classes, dataset, lambda_, teacher):
    """learning without forgetting"""
    save_name = f"lwf_{model}_{dataset}_{lambda_}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    trainer = LWFTransferLearningTrainer(
        _lambda=lambda_,
        teacher_model_path=str(settings.model_dir / teacher),
        model=get_model(model, num_classes),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")


@cli.command()
@click.option("-m", "--model", type=click.Choice(SupportParsevalModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
@click.option("-b", "--beta", type=float,
              default=1e-3, show_default=True, help="penalization rate of constrain")
@click.option("-k", "--k", type=int, required=True,
              help="trainable blocks from last")
@click.option("-t", "--teacher", type=str, required=True,
              help="filename of teacher model")
def ptl(model, num_classes, dataset, beta, k, teacher):
    """parseval transform learning"""
    save_name = f"bn_freeze_ptl_{model}_{dataset}_{beta}_{k}_from_{teacher}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")

    trainer = ParsevalTransferLearningTrainer(
        beta=beta,
        k=k,
        teacher_model_path=str(settings.model_dir / teacher),
        model=get_model(model, num_classes, k),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")


@cli.command()
@click.option("-m", "--model", type=click.Choice(SupportModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
@click.option("-k", "--k", type=int, required=True,
              help="trainable blocks from last")
@click.option("-t", "--teacher", type=str, required=True,
              help="filename of teacher model")
@click.option("-fb", "--freeze-bn", is_flag=True, help="freeze bn layer", show_default=True)
@click.option("-rs", "--reuse-statistic", is_flag=True, help="reuse statistic", show_default=True)
@click.option("-rts", "--reuse-teacher-statistic", is_flag=True, help="reuse teacher statistic", show_default=True)
def bntl(model, num_classes, dataset, k, teacher, freeze_bn, reuse_statistic, reuse_teacher_statistic):
    """normal transfer learning with batch norm operations"""
    save_name = f"bntl_{model}_{dataset}_{k}_{teacher}" \
                f"{'_fb' if freeze_bn else ''}" \
                f"{'_rs' if reuse_statistic else ''}" \
                f"{'_rts' if reuse_teacher_statistic else ''}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")

    trainer = BNTransferLearningTrainer(
        k=k,
        teacher_model_path=str(settings.model_dir / teacher),
        model=get_model(model, num_classes, k),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth",
        freeze_bn=freeze_bn,
        reuse_statistic=reuse_statistic,
        reuse_teacher_statistic=reuse_teacher_statistic
    )
    trainer.train(f"{settings.model_dir / save_name}")




@cli.command()
@click.option("-m", "--model", type=click.Choice(SupportNormalModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
@click.option("-k", "--k", type=int, required=True,
              help="trainable blocks from last")
@click.option("-st", "--state_dict", type=str, required=True,
              help="filename of state dict for model to be retrained")
def nr(model, num_classes, dataset, k, state_dict):
    """normal retrain"""
    save_name = f"nr_{model}_{dataset}_{k}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    model = get_model(model, num_classes, k)
    model.load_state_dict(torch.load(str(settings.model_dir / state_dict)))
    trainer = RetrainTrainer(
        k=k,
        model=model,
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")


@cli.command()
@click.option("-m", "--model", type=click.Choice(SupportParsevalModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
@click.option("-k", "--k", type=int, required=True,
              help="trainable blocks from last")
@click.option("-b", "--beta", type=float,
              default=1e-3, show_default=True, help="penalization rate of constrain")
@click.option("-st", "--state_dict", type=str, required=True,
              help="filename of state dict for model to be retrained")
def pr(model, num_classes, dataset, k, beta, state_dict):
    """parseval retrain"""
    save_name = f"pr_{model}_{dataset}_{k}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    model = get_model(model, num_classes, k)
    model.load_state_dict(torch.load(str(settings.model_dir / state_dict)))
    trainer = ParsevalRetrainTrainer(
        beta=beta,
        k=k,
        model=model,
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")


@cli.command()
@click.option("-m", "--model", type=click.Choice(SupportNormalModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
def nt(model, num_classes, dataset):
    """normal train"""
    save_name = f"nt_{model}_{dataset}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    trainer = NormalTrainer(
        model=get_model(model, num_classes),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")


@cli.command()
@click.option("-m", "--model", type=click.Choice(SupportModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
@click.option("--random_init/--no_random_init", default=True,
              show_default=True, help="PGD/BIM")
@click.option("-e", "--epsilon", type=float, default=8 / 255,
              show_default=True, help="epsilon")
@click.option("-ss", "--step_size", type=float, default=2 / 255,
              show_default=True, help="step size")
@click.option("-ns", "--num_steps", type=int, default=7,
              show_default=True, help="num steps")
def at(model, num_classes, dataset, random_init, epsilon, step_size, num_steps):
    """adversarial train"""
    save_name = f"at_{model}_{dataset}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    params = {
        "random_init": random_init,
        "epsilon": epsilon,
        "step_size": step_size,
        "num_steps": num_steps,
        "dataset_name": dataset
    }
    trainer = ADVTrainer(
        model=get_model(model, num_classes),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        attacker=LinfPGDAttack,
        params=params,
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")


@cli.command()
@click.option("-m", "--model", type=click.Choice(SupportModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
@click.option("--random_init/--no_random_init", default=True,
              show_default=True, help="PGD/BIM")
@click.option("-e", "--epsilon", type=float, default=8 / 255,
              show_default=True, help="epsilon")
@click.option("-ss", "--step_size", type=float, default=2 / 255,
              show_default=True, help="step size")
@click.option("-ns", "--num_steps", type=int, default=7,
              show_default=True, help="num steps")
@click.option("-k", "--k", type=int, required=True,
              help="kth(from last) layer norm will be used in loss")
@click.option("-l", "--lambda_", type=float, required=True,
              help="penalization rate of layer norm")
def cartl(model, num_classes, dataset, random_init, epsilon, step_size, num_steps, k, lambda_):
    """Cooperative Adversarially-Robust TransferLearning"""
    save_name = f"cartl_{model}_{dataset}_{k}_{lambda_}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    params = {
        "random_init": random_init,
        "epsilon": epsilon,
        "step_size": step_size,
        "num_steps": num_steps,
        "dataset_name": dataset
    }
    trainer = RobustPlusSingularRegularizationTrainer(
        k=k,
        _lambda=lambda_,
        model=get_model(model, num_classes),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        attacker=LinfPGDAttack,
        params=params,
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")


cli = cli()
