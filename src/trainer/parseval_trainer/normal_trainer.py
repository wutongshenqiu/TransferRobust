from typing import Tuple

import torch
from torch.utils.data import DataLoader

from ..base_trainer import BaseTrainer
from src.networks import WRN34Block
from src.utils import logger
from .mixins import ParsevalConstrainMixin
from src.networks import parseval_normal_wrn34_10, SupportedWideResnetType


class ParsevalNormalTrainer(BaseTrainer, ParsevalConstrainMixin):

    def __init__(self, beta: float, model: SupportedWideResnetType, train_loader: DataLoader,
                 test_loader: DataLoader, checkpoint_path: str = None):
        """initialize retrain trainer

        Args:
            beta: retraction parameter
        """
        super().__init__(model, train_loader, test_loader, checkpoint_path)
        self._blocks = WRN34Block(model)
        self.gather_constrain_layers(17, ignore_first_conv=False)

        logger.debug(f"beta: {beta}")
        self._beta = beta

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        inputs, labels = inputs.to(self._device), labels.to(self._device)
        self.optimizer.zero_grad()

        outputs = self.model(inputs)

        constrain_term = self.sum_layers_constrain()
        # logger.debug(f"batch constrain: {constrain_term}")
        loss = self.criterion(outputs, labels) + (self._beta / 2) * constrain_term
        loss.backward()
        self.optimizer.step()

        batch_training_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        batch_running_loss = loss.item()

        return batch_running_loss, batch_training_acc


if __name__ == '__main__':
    from src.utils import get_cifar_train_dataloader
    from src.utils import get_cifar_test_dataloader

    parseval_retrain_trainer = ParsevalNormalTrainer(
        beta=0.0003,
        model=parseval_normal_wrn34_10(num_classes=10),
        train_loader=get_cifar_train_dataloader(),
        test_loader=get_cifar_test_dataloader(),
    )
