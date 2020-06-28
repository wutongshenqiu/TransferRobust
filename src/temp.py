from utils import get_cifar_training_dataloader, get_cifar_testing_dataloader
import torchvision.transforms as transforms
import torch

# default mean of cifar100
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
# default std of cifar100
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


if __name__ == "__main__":
    normalize_testing_loader = get_cifar_testing_dataloader("cifar100")
    testing_loader = get_cifar_testing_dataloader("cifar100", normalize=False)

    normalize = transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)

    for data, label in testing_loader:
        print(len(data[:]))
        d = label[0]
        break



    pass


