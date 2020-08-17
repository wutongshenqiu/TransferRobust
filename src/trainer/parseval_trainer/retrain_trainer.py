"""parseval training

constrains:
    1. residual layer: f(x) + x -> 0.5x + 0.5f(x)
    2. fully connect layer: ||W * W^T - I||^2
    3. convolution layer: ||W * W^T - I / scaling||^2
"""

from typing import Tuple

import torch
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader

from ..retrain_trainer import RetrainTrainer

from src.utils import logger
from .mixins import ParsevalTrainerMixin
from src.networks import parseval_retrain_wrn34_10


class ParsevalRetrainTrainer(RetrainTrainer, ParsevalTrainerMixin):
    def __init__(self, beta: float, k: int, model: Module, train_loader: DataLoader,
                 test_loader: DataLoader, checkpoint_path: str = None):
        """initialize retrain trainer

        Args:
            beta: retraction parameter
            k: the last k blocks which will be retrained
        """
        super(ParsevalRetrainTrainer, self).__init__(k, model, train_loader, test_loader, checkpoint_path)
        self._gather_constrain_layers(k)
        self._beta = beta

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        inputs, labels = inputs.to(self._device), labels.to(self._device)
        self.optimizer.zero_grad()

        outputs = self.model(inputs)

        constrain_term = self.sum_layers_constrain()
        logger.debug(f"batch constrain: {constrain_term}")
        loss = self.criterion(outputs, labels) + (self._beta / 2) * constrain_term
        loss.backward()
        self.optimizer.step()

        batch_training_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        batch_running_loss = loss.item()

        return batch_running_loss, batch_training_acc


if __name__ == '__main__':
    from src.utils import get_cifar_train_dataloader
    from src.utils import get_cifar_test_dataloader

    parseval_retrain_trainer = ParsevalRetrainTrainer(
        beta=0.0003,
        k=6,
        model=parseval_retrain_wrn34_10(k=6, num_classes=10),
        train_loader=get_cifar_train_dataloader(),
        test_loader=get_cifar_test_dataloader(),
    )
