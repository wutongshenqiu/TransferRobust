from typing import Dict

import torch
from torch.utils.data import DataLoader

from ..adv_trainer import ADVTrainer
from src.networks import WRN34Block
from ..mixins import InitializeTensorboardMixin
from src.utils import logger
from src.networks import SupportedWideResnetType


class RobustPlusAllRegularizationTrainer(ADVTrainer, InitializeTensorboardMixin):
    def __init__(self, _lambda: float, model: SupportedWideResnetType, train_loader: DataLoader,
                 test_loader: DataLoader, attacker, params: Dict, checkpoint_path: str = None):
        super().__init__(model, train_loader, test_loader,
                         attacker, params, checkpoint_path)
        self._blocks = WRN34Block(model)
        self._register_forward_hook_to_all_block()
        self._hooked_features_list = []
        self._lambda = _lambda

        self.summary_writer = self.init_writer()

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
        inputs, labels = inputs.to(self._device), labels.to(self._device)

        adv_inputs = self._gen_adv(inputs, labels)
        adv_outputs = self.model(adv_inputs)
        clean_outputs = self.model(inputs)

        regularization_term = self._calculate_regularization()
        l_term = self.criterion(adv_outputs, labels)
        loss = l_term + self._lambda * regularization_term

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        batch_training_acc = (clean_outputs.argmax(dim=1) == labels).float().mean().item()
        batch_running_loss = loss.item()

        return batch_running_loss, batch_training_acc

    def _calculate_regularization(self) -> torch.Tensor:
        def calculate_kth_regularization(r_adv: torch.Tensor, r_clean: torch.Tensor) -> torch.Tensor:
            return torch.norm(
                (r_adv - r_clean).view(r_adv.shape[0], -1),
                dim=1
            ).sum()

        sum_regularization = sum(map(
            calculate_kth_regularization,
            self._hooked_features_list[:17],
            self._hooked_features_list[17:]
        ))

        self._hooked_features_list.clear()

        return sum_regularization

    def _register_forward_hook_to_all_block(self):
        for k in range(1, 18):
            logger.debug(f"register hook to the last layer of {k}th block")
            block = getattr(self._blocks, f"block{k}")
            block.register_forward_hook(self._get_layer_outputs)

    def _get_layer_outputs(self, layer, inputs, outputs):
        if self.model.training:
            self._hooked_features_list.append(outputs.clone().detach())


if __name__ == '__main__':
    from src.networks import wrn34_10
    from src.utils import get_cifar_test_dataloader, get_cifar_train_dataloader
    from src.attack import LinfPGDAttack, attack_params

    model = wrn34_10(num_classes=100)
    save_path = f"123123"
    trainer = RobustPlusRegularizationTrainer(
        _lambda=0.05,
        model=model,
        train_loader=get_cifar_train_dataloader("cifar100"),
        test_loader=get_cifar_test_dataloader("cifar100"),
        attacker=LinfPGDAttack,
        params=attack_params.get("LinfPGDAttack"),
        checkpoint_path=f"../checkpoint/{save_path}.pth",
    )

    trainer.train(f"../trained_models/{save_path}")
