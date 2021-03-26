import torch

from src import settings
from src.utils import logger
from src.trainer import RobustPlusSpectrumNormTrainer

from src.attack import LinfPGDAttack
from src.cli.utils import get_test_dataset, get_train_dataset, get_model


def sn_at(model, num_classes, dataset, power_iter, beta_norm, random_init, epsilon, step_size, num_steps):
    save_name = f"snat_{model}_{dataset}_{power_iter}_{beta_norm}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")

    params  = {
        "random_init": random_init,
        "epsilon": epsilon,
        "step_size": step_size,
        "num_steps": num_steps,
        "dataset_name": dataset
    }

    trainer = RobustPlusSpectrumNormTrainer(
        model=get_model(model=model, num_classes=num_classes),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        attacker=LinfPGDAttack,
        params=params,
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth",
        beta_norm=beta_norm,
        power_iter=power_iter,
    )
    trainer.train(f"{settings.model_dir / save_name}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-n", "--num_classes", type=int)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-e", "--epsilon", type=float, default=8/255)
    parser.add_argument("--step-size", type=float, default=2/255)
    parser.add_argument("--num-steps", type=int, default=7)
    parser.add_argument("--random-init", action="store_false") #default value is True
    parser.add_argument("--power-iter", type=int, default=1)
    parser.add_argument("--beta-norm", type=float, default=1.0)

    args = parser.parse_args()

    sn_at(model=args.model, num_classes=args.num_classes, dataset=args.dataset, power_iter=args.power_iter, beta_norm=args.beta_norm,
             random_init=args.random_init, epsilon=args.epsilon, step_size=args.step_size, num_steps=args.num_steps)


