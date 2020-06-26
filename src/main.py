import argparse
from hyperparameters import *
from train import Trainer
from models import resnet
from utils import get_test_dataloader, get_training_dataloader
from attack import FGSMAttack, PGDAttack


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="a cli that supports train/tl/attack")
    parser.add_argument("--pattern", required=True, help="use train/tl/attack to determine the pattern")
    parser.add_argument("--dataset", required=True, help="choose the dataset, supporting cifar100 or cifar10")
    parser.add_argument("--save_path", required=True, help="the saving path of model")
    # parser.add_argument("--model", help="the name of training model", default="resnet50")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="training device")
    parser.add_argument("--lr", default=DEFAULT_LR, help="learning rate", type=float)
    parser.add_argument("--momentum", default=DEFAULT_MOMENTUM, help="momentum", type=float)
    parser.add_argument("--epochs", default=DEFAULT_EPOCHS, help="training epochs", type=int)
    parser.add_argument("--batch", default=DEFAULT_BATCH_SIZE, help="the size of each batch", type=int)
    parser.add_argument("--num_worker", default=DEFAULT_NUM_WORKER, help="num_workers", type=int)
    parser.add_argument("--gamma", default=0.2, help="decrease rate when reach milestones", type=float)
    parser.add_argument("--phases", default=WARM_PHASES, help="epochs for warming model", type=int)

    parser.add_argument("--old", help="old model path")
    parser.add_argument("--attacker", help="attackers, now support FGSMAttack and PGDAttack")
    parser.add_argument("--epsilon", help="epsilons", type=float)

    # distributeddataparallel
    # parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()
    if args.pattern in ("train", "attack"):
        train_loader = get_training_dataloader(args.dataset, args.batch, args.num_worker)
        test_loader = get_test_dataloader(args.dataset, args.batch, args.num_worker)
        if args.dataset == "cifar100":
            model = resnet.resnet50()
        elif args.dataset == "cifar10":
            model = resnet.resnet50(num_classes=10)
        else:
            raise Exception

        trainer = Trainer(model, train_loader, test_loader, args.device, args.lr,
                          args.momentum, args.epochs, args.batch, DEFAULT_PARALLELISM,
                          MILESTONES, args.gamma, args.phases)
        if args.pattern == "train":
            trainer.train(args.save_path)
        else:
            if args.dataset == "cifar10":
                raise Exception("dataset must be cifar100")
            if not args.attacker and args.epsilons:
                raise ValueError("attacker and params must be specify")
            if args.attacker == "fgsm":
                attacker = FGSMAttack
            elif args.attacker == "pgd":
                attacker = PGDAttack
            else:
                raise NotImplemented
            trainer.train(args.save_path, attack=True, attacker=attacker, params={"epsilon": args.epsilon})

    elif args.pattern == "tl":
        if not args.old:
            raise ValueError("original model must be specify")
        train_loader = get_training_dataloader("cifar10", args.batch, args.num_worker)
        test_loader = get_test_dataloader("cifar10", args.batch, args.num_worker)
        Trainer.train_tl(args.old, args.save_path, train_loader, test_loader, args.device)







