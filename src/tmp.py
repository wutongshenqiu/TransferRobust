import torch
import torch.nn as nn
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
from typing import Callable, Any


DATA_DIR = r"~/dataset"
DEVICE = "cuda"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_cifar10_test_dataloader(batch_size=128, num_workers=4,
                                shuffle=False, normalize=True):

    _data = torchvision.datasets.CIFAR10

    mean, std = ((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))

    compose_list = [
        transforms.ToTensor(),
    ]
    if normalize:
        compose_list.append(transforms.Normalize(mean, std))
    transform_test = transforms.Compose(compose_list)
    test = _data(root=os.path.join(DATA_DIR, "CIFAR"), train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader


def get_mnist_train_dataloader(batch_size=128, num_workers=4,
                               shuffle=True, normalize=True):
    compose_list = [
        # resize original mnist size(28 * 28) to 32 * 32
        # transforms.Resize(32),
        transforms.ToTensor(),
    ]
    if normalize:
        mean, std = (0.1307,), (0.3081,)
        compose_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(compose_list)

    train_data = torchvision.datasets.MNIST(root=os.path.join(DATA_DIR, "MNIST"), train=True,
                                            download=True, transform=transform)
    train_loader = DataLoader(train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return train_loader


def clamp(t: Tensor, lower_limit, upper_limit):
    return torch.max(torch.min(t, upper_limit), lower_limit)


def get_mnist_test_dataloader(batch_size=128, num_workers=4,
                              shuffle=False, normalize=True):
    compose_list = [
        # transforms.Resize(32),
        transforms.ToTensor(),
    ]
    if normalize:
        mean, std = (0.1307,), (0.3081,)
        compose_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(compose_list)

    test_data = torchvision.datasets.MNIST(root=os.path.join(DATA_DIR, "MNIST"), train=False,
                                           download=True, transform=transform)
    test_loader = DataLoader(test_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader


class LinfPGDAttack:

    def __init__(self, model: torch.nn.Module, clip_min=0, clip_max=1,
                 random_init: int = 1, epsilon=8/255, step_size=2/255, num_steps=20,
                 loss_function: Callable[[Any], Tensor] = nn.CrossEntropyLoss(),
                 ):
        dataset_mean, dataset_std = ((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        mean = torch.tensor(dataset_mean).view(3, 1, 1).to(DEVICE)
        std = torch.tensor(dataset_std).view(3, 1, 1).to(DEVICE)

        clip_max = ((clip_max - mean) / std)
        clip_min = ((clip_min - mean) / std)
        epsilon = epsilon / std
        step_size = step_size / std

        self.min = clip_min
        self.max = clip_max
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.random_init = random_init
        self.num_steps = num_steps
        self.loss_function = loss_function

    def random_delta(self, delta: Tensor) -> Tensor:
        delta.uniform_(-1, 1)
        delta = delta * self.epsilon

        return delta

    def calc_perturbation(self, x: Tensor, target: Tensor) -> Tensor:
        delta = torch.zeros_like(x)
        if self.random_init:
            delta = self.random_delta(delta)
        xt = x + delta
        xt.requires_grad = True

        for it in range(self.num_steps):
            y_hat = self.model(xt)
            loss = self.loss_function(y_hat, target)

            self.model.zero_grad()
            loss.backward()

            grad_sign = xt.grad.detach().sign()
            xt.data = xt.detach() + self.step_size * grad_sign
            xt.data = clamp(xt - x, -self.epsilon, self.epsilon) + x
            xt.data = clamp(xt.detach(), self.min, self.max)

            xt.grad.data.zero_()

        return xt


if __name__ == '__main__':
    attack_params = {
        "cifar100": {
            "random_init": 1,
            "epsilon": 8 / 255,
            "step_size": 2 / 255,
            "num_steps": 7,
        },
        "svhn": {
            "random_init": 1,
            "epsilon": 0.3,
            "step_size": 0.01,
            "num_steps": 40,
        }
    }
    model_path = ""
    model = Net()
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )
    attacker = LinfPGDAttack(model, **attack_params["cifar100"])

    cifar10_test_loader = get_cifar10_test_dataloader(batch_size=10)

    for inputs, labels in cifar10_test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        print(f"origin labels: {labels}")
        adv_inputs = attacker.calc_perturbation(inputs, labels)
        with torch.no_grad():
            _, y_hats = model(adv_inputs).max(1)
            print(f"adv labels: {y_hats}")
        break

