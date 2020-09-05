from typing import Dict

import torch
from torch.utils.data import DataLoader

from ..adv_trainer import ADVTrainer
from ..mixins import InitializeTensorboardMixin
from src.utils import logger
from src.networks import SupportedAllModuleType, make_blocks


class RobustPlusSingularRegularizationTrainer(ADVTrainer, InitializeTensorboardMixin):
    def __init__(self, k: int, _lambda: float, model: SupportedAllModuleType, train_loader: DataLoader,
                 test_loader: DataLoader, attacker, params: Dict, checkpoint_path: str = None):
        super().__init__(model, train_loader, test_loader,
                         attacker, params, checkpoint_path)
        self._blocks = make_blocks(model)
        self._register_forward_hook_to_k_block(k)
        self._hooked_features_list = []
        self._lambda = _lambda

        self.summary_writer = self.init_writer()

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
        inputs, labels = inputs.to(self._device), labels.to(self._device)

        adv_inputs = self._gen_adv(inputs, labels)
        adv_outputs = self.model(adv_inputs)
        clean_outputs = self.model(inputs)

        r_adv = self._hooked_features_list[0]
        r_clean = self._hooked_features_list[1]
        # so stupid!
        regularization_term = self._lambda * torch.norm(
            (r_adv - r_clean).view(r_adv.shape[0], -1),
            dim=1
        ).sum()
        # logger.debug(f"d_loss: {regularization_term}")

        self._hooked_features_list.clear()

        l_term = self.criterion(adv_outputs, labels)
        # logger.debug(f"l_loss: {l_term}")

        loss = l_term + regularization_term

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        batch_training_acc = (clean_outputs.argmax(dim=1) == labels).float().mean().item()
        batch_running_loss = loss.item()

        return batch_running_loss, batch_training_acc

    def _register_forward_hook_to_k_block(self, k):
        total_blocks = self._blocks.get_total_blocks()
        assert 1 <= k <= total_blocks
        logger.debug(f"model total blocks: {total_blocks}")
        logger.debug(f"register hook to the last layer of {k}th block from last")
        block = getattr(self._blocks, f"block{total_blocks+1-k}")
        block.register_forward_hook(self.get_layer_outputs)

    def get_layer_outputs(self, layer, inputs, outputs):
        if self.model.training:
            self._hooked_features_list.append(outputs.clone().detach())
