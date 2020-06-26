from train import Trainer
from utils import get_cifar_training_dataloader, get_cifar_testing_dataloader
from networks.wrn import wrn34_10


if __name__ == "__main__":
    model = wrn34_10()

    print(model.__name__)

    import torch

    print(str(torch.device("cuda: 0")))
