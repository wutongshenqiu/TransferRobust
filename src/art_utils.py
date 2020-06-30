from art.attacks.evasion import ProjectedGradientDescent
from art import attacks
from art.estimators.classification import PyTorchClassifier

from typing import Tuple, Dict

import numpy as np

from config import settings
from utils import CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD, CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD

import torch.nn as nn
import torch


def init_preprocessing(dataset: str):
    if dataset == "cifar100":
        mean = CIFAR100_TRAIN_MEAN
        std = CIFAR100_TRAIN_STD
    elif dataset == "cifar10":
        mean = CIFAR10_TRAIN_MEAN
        std = CIFAR10_TRAIN_STD

    return (np.asarray(mean).reshape((3, 1, 1)),
            np.asarray(std).reshape((3, 1, 1)))


def init_classifier(
        *,
        model,
        preprocessing=init_preprocessing("cifar100"),
        input_shape: Tuple[int] = (3, 32, 32),
        nb_classes: int = 100,
        clip_values=(0, 1),
):
    classifier = PyTorchClassifier(
        model=model,
        preprocessing=preprocessing,
        input_shape=input_shape,
        nb_classes=nb_classes,
        clip_values=clip_values,
        loss=getattr(nn, settings.criterion)()
    )
    classifier.set_learning_phase(False)
    classifier.fit()

    return classifier


def init_attacker(classifier: PyTorchClassifier, attacker_name: str, params: Dict):
    return getattr(attacks.evasion, attacker_name)(
        estimator=classifier,
        **params
    )


def gen_adv(attacker, x_test: torch.Tensor) -> torch.Tensor:
    x_test = x_test.cpu().numpy()
    return torch.from_numpy(attacker.generate(x=x_test))


attack_params = {
    "ProjectedGradientDescent": {
        "eps": 8 / 255,
        "eps_step": 2 / 255,
        "batch_size": settings.batch_size,
        "max_iter": 7,
        "num_random_init": 1
    }
}