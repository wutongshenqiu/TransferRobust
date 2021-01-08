from os import EX_SOFTWARE
from src.cli.utils import get_model, get_test_dataset, get_train_dataset
import torch

from src import settings
from src.utils import logger
from src.trainer import RobustPlusWassersteinTrainer

from src.attack import LinfPGDAttack
from src.networks import wrn34_10
from src.utils import get_cifar_test_dataloader, get_cifar_train_dataloader


def wd_fdm(model, num_classes, _lambda, lr_estimator, dataset, k, random_init, epsilon, step_size, num_steps, pretrained=None):
    save_name = f"wd_fdm_{model}_{dataset}_{k}_{_lambda}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")

    params  = {
        "random_init": random_init,
        "epsilon": epsilon,
        "step_size": step_size,
        "num_steps": num_steps,
        "dataset_name": dataset
    }

    model = get_model(model, num_classes, k)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained, map_location=settings.device))

    trainer = RobustPlusWassersteinTrainer(
        k=k,
        model=model,
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        attacker=LinfPGDAttack,
        params=params,
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth",
        lr_estimator=lr_estimator,
        lambda_=_lambda
    )

    trainer.train(f"{settings.model_dir / save_name}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-n", "--num_classes", type=int, required=True)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-e", "--epsilon", type=float, default=8/255)
    parser.add_argument("--step-size", type=float, default=2/255)
    parser.add_argument("--num-steps", type=int, default=7)
    parser.add_argument("-k", "--k", type=int, required=True)
    parser.add_argument("-l", "--lambda_", type=float, required=True)
    parser.add_argument("--random-init", action="store_false") #default value is True
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("-lr-estimator", type=float, default=0.0001)

    args = parser.parse_args()

    wd_fdm(
        model=args.model,
        num_classes=args.num_classes,
        _lambda=args.lambda_,
        lr_estimator=args.lr_estimator,
        dataset=args.dataset,
        k=args.k,
        random_init=args.random_init,
        epsilon=args.epsilon,
        step_size=args.step_size,
        num_steps=args.num_steps,
        pretrained=args.pretrained
    )