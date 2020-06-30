import sys
import os
import json
import random
import time
from pprint import pprint

from typing import Tuple, Dict

from art.attacks.evasion import DeepFool, FastGradientMethod, ProjectedGradientDescent, CarliniL2Method, PixelAttack, Attack
from art.classifiers import PyTorchClassifier
import torch
import torch.nn as nn
import numpy as np

from utils import test_model, get_mnist_testing_data, get_cifar_testing_dataloader


class MyPyTorchClassifier(PyTorchClassifier):

    def __init__(self, *,
                 model: nn.Module,
                 input_shape: Tuple,
                 nb_classes: int,
                 loss=nn.CrossEntropyLoss(),
                 optimizer=None,
                 **kwargs
                 ):
        super().__init__(
            model=model,
            input_shape=input_shape,
            nb_classes=nb_classes,
            loss=loss,
            optimizer=optimizer,
            **kwargs
        )


def test_attack(model: nn.Module, x_test: np.ndarray, y_test: np.ndarray,
         attacker: type(Attack), attacker_params: Dict, nb_classes=10, **kwargs) -> float:

    start = time.time()

    classifier = MyPyTorchClassifier(
        model=model,
        input_shape=(3, 32, 32),
        nb_classes=nb_classes,
        clip_values=(0, 1),
        **kwargs
    )

    classifier.set_learning_phase(False)
    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    # Step 6: Generate adversarial test examples
    print(f"generate adversarial examples using attack: {attacker.__name__}")
    print("parameters:")
    pprint(attacker_params)
    attack = attacker(classifier=classifier, **attacker_params)
    x_test_adv = attack.generate(x=x_test)

    # Step 7: Evaluate the ART classifier on adversarial test examples

    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

    end = time.time()
    print(f"costing time: {end-start:.2f}s")
    print("="*100)

    return accuracy

