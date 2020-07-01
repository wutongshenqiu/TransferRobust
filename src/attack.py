from __future__ import print_function
from typing import Any, Callable, Union

import torch
from torch import Tensor
import torch.nn as nn
import numpy as np


attack_params = {
    "LinfPGDAttack": {
        "random_init": 1,
        "epsilon": 8/255,
        "step_size": 2/255,
        "num_steps": 7
    }
}


class LinfPGDAttack:

    def __init__(self, model: torch.nn.Module, clip_min=0, clip_max=1,
                 random_init: int = True, epsilon=8/255, step_size=2/255, num_steps=20,
                 loss_function: Callable[[Any], Tensor] = nn.CrossEntropyLoss()
                 ):

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
            xt.data = torch.clamp(xt - x, -self.epsilon, self.epsilon) + x
            xt.data = torch.clamp(xt.detach(), self.min, self.max)

            xt.grad.data.zero_()

        return xt

    def print_parameters(self):
        print(f"{self.__dict__}")


if __name__ == '__main__':
    pass

