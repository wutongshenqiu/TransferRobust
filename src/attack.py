from typing import Any, Callable, Dict

import torch
from torch import Tensor
import torch.nn as nn

from . import settings
from .utils import logger, get_mean_and_std, clamp, evaluate_accuracy


attack_params = {
    "cifar100": {
        "random_init": 1,
        "epsilon": 8/255,
        "step_size": 2/255,
        "num_steps": 7,
    },
    "svhn": {
        "random_init": 1,
        "epsilon": 0.3,
        "step_size": 0.01,
        "num_steps": 40,
    }
}


class LinfPGDAttack:

    def __init__(self, model: torch.nn.Module, clip_min=0, clip_max=1,
                 random_init: int = 1, epsilon=8/255, step_size=2/255, num_steps=20,
                 loss_function: Callable[[Any], Tensor] = nn.CrossEntropyLoss(),
                 dataset_name: str = settings.dataset_name
                 ):
        dataset_mean, dataset_std = get_mean_and_std(dataset_name)
        mean = torch.tensor(dataset_mean).view(3, 1, 1).to(settings.device)
        std = torch.tensor(dataset_std).view(3, 1, 1).to(settings.device)

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

    def print_parameters(self):
        params = {
            "min": self.min,
            "max": self.max,
            "epsilon": self.epsilon,
            "step_size": self.step_size,
            "num_steps": self.num_steps,
            "random_init": self.random_init,
        }
        params_str = "\n".join([": ".join(map(str, item)) for item in params.items()])
        logger.info(f"using attack: {type(self).__name__}")
        logger.info(f"attack parameters: \n{params_str}")


def test_attack(model: nn.Module, test_loader, attacker, params: Dict, device: str = settings.device) -> float:
    normal_acc = evaluate_accuracy(model, test_loader, device)
    logger.info(f"normal accuracy: {normal_acc}")
    model.eval()
    _attacker = attacker(model, **params)
    _attacker.print_parameters()

    correct = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        adv_inputs = _attacker.calc_perturbation(inputs, labels)
        model.zero_grad()
        with torch.no_grad():
            _, y_hats = model(adv_inputs).max(1)
            match = (y_hats == labels)
            correct += len(match.nonzero())

    adversarial_accuracy = correct / len(test_loader.dataset)
    logger.info(f"adversarial accuracy: {100 * adversarial_accuracy:.3f}%")

    model.train()

    return adversarial_accuracy


if __name__ == '__main__':
    from src.networks import parseval_retrain_wrn34_10, wrn34_10
    from .utils import get_cifar_test_dataloader, get_cifar_train_dataloader
    import time
    import json


    params = {
        "random_init": 1,
        "epsilon": 8/255,
        "step_size": 2/255,
        "num_steps": 20,
        "dataset_name": "cifar100",
    }
    
    result = {}
    # model = wrn34_10(num_classes=10)
    _lambda = 1
    map_beta = {1e-3: "1e-3", 2e-3: "2e-3", 3e-4: "3e-4", 6e-4: "6e-4"}
    
    logger.change_log_file(settings.log_dir / f"normalization_cifar100_tl_pgd7_attack.log")
    test_loader = get_cifar_test_dataloader("cifar10") 
    model = wrn34_10(num_classes=10)
    for k in range(1, 18):
        model_path = f"./trained_models/normalization_cifar100_tl_pgd7_blocks{k}-best"
        
        logger.debug(f"load from `{model_path}`")
        model.load_state_dict(torch.load(model_path, map_location=settings.device))
        model.to(settings.device)
        start_time = time.perf_counter()
        acc = test_attack(model, test_loader, LinfPGDAttack, params)
        end_time = time.perf_counter()

        logger.info(f"costing time: {end_time-start_time:.2f} secs")
