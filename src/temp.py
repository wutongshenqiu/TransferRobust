from train import Trainer
from utils import get_cifar_training_dataloader, get_cifar_testing_dataloader
from models.wrn import wrn34_10


if __name__ == "__main__":
    Trainer.train_tl(
        origin_model_path="./trained_models/cifar100_wrn34_10-best",
        save_path="./trained_models/tl_cifar10_wrn34_10",
        train_loader=get_cifar_training_dataloader("cifar10"),
        test_loader=get_cifar_testing_dataloader("cifar10"),
        device="cuda: 0"
    )

