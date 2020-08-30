from typing import Dict, Any, Tuple
import os

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torch
from torch import Tensor
from torch.nn import Module
import numpy as np
import matplotlib.pyplot as plt
import json

from src import settings
from ..logging_utils import logger
from .dataset_utils import SubsetDataset

DATA_DIR = "~/dataset"

# default mean of cifar100
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
# default std of cifar100
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
# default mean of cifar10
CIFAR10_TRAIN_MEAN = (0.49139765, 0.48215759, 0.44653141)
# default std of cifar10
CIFAR10_TRAIN_STD = (0.24703199, 0.24348481, 0.26158789)

MNIST_TRAINING_STD = 0.3081
MNIST_TRAINING_MEAN = 0.1307


def clamp(t: Tensor, lower_limit, upper_limit):
    return torch.max(torch.min(t, upper_limit), lower_limit)


def get_mean_and_std(dataset: str = settings.dataset_name) -> Tuple[Tuple, Tuple]:
    if dataset == "cifar100":
        logger.debug(f"get mean and std of cifar100")
        mean = CIFAR100_TRAIN_MEAN
        std = CIFAR100_TRAIN_STD
    elif dataset == "cifar10":
        # logger.debug(f"get mean and std of cifar10")
        logger.warning("Using mean and std of cifar100 for dataset cifar10!")
        mean = CIFAR100_TRAIN_MEAN
        std = CIFAR100_TRAIN_STD
    else:
        raise ValueError(f'dataset "{dataset}" is not supported!')

    return mean, std


def get_subset_cifar_train_dataloader(partition_ratio: float, dataset=settings.dataset_name,
                                      batch_size=settings.batch_size, num_workers=settings.num_worker,
                                      shuffle=True, normalize=True):
    logger.info("load whole loader")
    whole_cifar_dataloader = get_cifar_train_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # false to keep same
        shuffle=False,
        normalize=normalize
    )
    subset_dataset = SubsetDataset(whole_cifar_dataloader, partition_ratio)

    return DataLoader(subset_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)


def get_cifar_train_dataloader(dataset=settings.dataset_name, batch_size=settings.batch_size,
                               num_workers=settings.num_worker, shuffle=True, normalize=True):
    if dataset == "cifar100":
        _data = torchvision.datasets.CIFAR100
        logger.info("load cifar100 train dataset")
    elif dataset == "cifar10":
        _data = torchvision.datasets.CIFAR10
        logger.info("load cifar10 train dataset")
    else:
        raise ValueError(f'dataset "{dataset}" is not supported!')

    mean, std = get_mean_and_std(dataset=dataset)
    logger.debug(f"dataset mean: {mean}, dataset std: {std}")

    compose_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ]
    if normalize:
        compose_list.append(transforms.Normalize(mean, std))
    transform_train = transforms.Compose(compose_list)

    train_dataset = _data(root=os.path.join(DATA_DIR, "CIFAR"), train=True, download=True,
                          transform=transform_train)

    train_loader = DataLoader(train_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return train_loader


def get_cifar_test_dataloader(dataset=settings.dataset_name, batch_size=settings.batch_size,
                              num_workers=settings.num_worker, shuffle=False, normalize=True):
    if dataset == "cifar100":
        _data = torchvision.datasets.CIFAR100
        logger.info("load cifar100 test dataset")
    elif dataset == "cifar10":
        _data = torchvision.datasets.CIFAR10
        logger.info("load cifar10 test dataset")
    else:
        raise ValueError(f'dataset "{dataset}" is not supported!')

    mean, std = get_mean_and_std(dataset=dataset)

    compose_list = [
        transforms.ToTensor(),
    ]
    if normalize:
        compose_list.append(transforms.Normalize(mean, std))
    transform_test = transforms.Compose(compose_list)
    test = _data(root=os.path.join(DATA_DIR, "CIFAR"), train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader


def get_mnist_train_data(batch_size=128, num_workers=4, normalize=True):
    compose_list = [
        transforms.ToTensor(),
    ]
    if normalize:
        compose_list.append(transforms.Normalize(MNIST_TRAINING_MEAN, MNIST_TRAINING_STD))
    transform = transforms.Compose(compose_list)

    train_data = torchvision.datasets.MNIST(root=os.path.join(DATA_DIR, "MNIST"), train=True,
                                            download=True, transform=transform)
    train_loader = DataLoader(train_data, shuffle=True, num_workers=num_workers, batch_size=batch_size)

    return train_loader


def get_mnist_test_data(batch_size=128, num_workers=4, normalize=True):
    compose_list = [
        transforms.ToTensor(),
    ]
    if normalize:
        compose_list.append(transforms.Normalize(MNIST_TRAINING_MEAN, MNIST_TRAINING_STD))
    transform = transforms.Compose(compose_list)

    test_data = torchvision.datasets.MNIST(root=os.path.join(DATA_DIR, "MNIST"), train=False,
                                           download=True, transform=transform)
    test_loader = DataLoader(test_data, shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return test_loader


def evaluate_accuracy(model: Module, test_loader: DataLoader, device: torch.device, debug=False) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            _, y_hats = model(inputs).max(1)
            match = (y_hats == labels)
            correct += len(match.nonzero())

    if debug:
        print(f"Testing: {len(test_loader.dataset)}")
        print(f"correct: {correct}")
        print(f"accuracy: {100 * correct / len(test_loader.dataset):.3f}%")

    model.train()

    return correct / len(test_loader.dataset)


def get_fc_out_features(model: Module):
    if not hasattr(model, "fc"):
        raise AttributeError("model doesn't have attribute as fully connected layer!")

    return model.fc.out_features


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = np.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = np.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = np.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def attack_loss(X, y, model: torch.nn.Module, attacker, eps=0.1, loss=torch.nn.CrossEntropyLoss()) -> torch.Tensor:
    y_hat = model(X)
    adv_y_hat = attacker(model, eps=eps).cal_perturbation(X, y)

    return loss(y_hat, y) + loss(adv_y_hat, y)


def grey_to_img(img: Tensor):
    # change (3, H, W) to (H, W, 3)
    size = img.shape
    if len(size) == 3:
        area = size[1] * size[2]
        red = img[0].reshape(area, 1)
        green = img[1].reshape(area, 1)
        blue = img[2].reshape(area, 1)
        new_img = np.hstack([red, green, blue]).reshape((size[1], size[2], 3))
    elif len(size) == 2:
        new_img = img
    else:
        raise NotImplementedError(f"img shape: {size} is not supported")
    plt.imshow(new_img)
    plt.axis("off")
    plt.show()


def load_json(json_path: str) -> Dict[Any, Any]:
    with open(json_path, "r", encoding="utf8") as f:
        return json.loads(f.read())

