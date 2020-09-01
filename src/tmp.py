from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torchvision
from torchvision.transforms import transforms

import os

DATA_DIR = "../../dataset"


def compute_mean_std(dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = np.dstack([dataset[i][0, :, :] for i in range(len(dataset))])
    data_g = np.dstack([dataset[i][1, :, :] for i in range(len(dataset))])
    data_b = np.dstack([dataset[i][2, :, :] for i in range(len(dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std


if __name__ == '__main__':
    mean = (0.4376817, 0.4437706, 0.4728039)
    std = (0.19803032, 0.20101574, 0.19703609)
    # this will apply to data when load from dataloader
    compose_list = [
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    transform = transforms.Compose(compose_list)
    # cifar10_dataset = torchvision.datasets.CIFAR10(root=os.path.join(DATA_DIR, "CIFAR"), train=True,
    #                                             download=True, transform=transform)
    # cifar10_dataloader = DataLoader(cifar10_dataset, batch_size=128, num_workers=0, shuffle=False)

    # svhn_dataset = torchvision.datasets.SVHN(root=os.path.join(DATA_DIR, "SVHN"), split="train",
    #                                          download=True, transform=transform)
    # svhn_dataloader = DataLoader(svhn_dataset, batch_size=128, num_workers=0, shuffle=False)

    mnist_dataset = torchvision.datasets.MNIST(root=os.path.join(DATA_DIR, "MNIST"), train=True,
                                               download=True, transform=transform)
    mnist_dataloader = DataLoader(mnist_dataset, batch_size=128, num_workers=0, shuffle=False)

    inputs_tensor_list = []
    for inputs, _ in mnist_dataloader:
        inputs_tensor_list.append(inputs)

    dataset = torch.cat(inputs_tensor_list, dim=0)
    print(compute_mean_std(dataset))
