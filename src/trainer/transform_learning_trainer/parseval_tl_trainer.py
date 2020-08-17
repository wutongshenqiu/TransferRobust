import torch
from torch.utils.data import DataLoader

from typing import Tuple

from .mixins import ReshapeTeacherFCLayerMixin
from ..parseval_trainer import ParsevalConstrainMixin
from .tl_trainer import TransformLearningTrainer
from src.networks import SupportedModuleType
from src.utils import logger


class ParsevalTransformLearningTrainer(TransformLearningTrainer, ParsevalConstrainMixin):

    def __init__(self, beta: float, k: int, teacher_model_path: str,
                 model: SupportedModuleType, train_loader: DataLoader,
                 test_loader: DataLoader, checkpoint_path: str = None):
        super().__init__(k, teacher_model_path, model, train_loader, test_loader, checkpoint_path)

        self.gather_constrain_layers(k)
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
    from src.networks import wrn34_10
    from src.utils import get_cifar_test_dataloader, get_cifar_train_dataloader

    trainer = ParsevalTransformLearningTrainer(
        beta=0.0003,
        k=6,
        teacher_model_path="./trained_models/cifar100_pgd7_train-best",
        model=wrn34_10(num_classes=10),
        train_loader=get_cifar_train_dataloader("cifar10"),
        test_loader=get_cifar_test_dataloader("cifar10"),
    )