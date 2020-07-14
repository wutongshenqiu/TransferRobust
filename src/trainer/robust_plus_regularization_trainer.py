from typing import Dict

import torch
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader

from trainer import ADVTrainer, WRN34Block


class RobustPlusRegularizationTrainer(ADVTrainer):
    def __init__(self, k: int, _lambda: float, model: Module, train_loader: DataLoader,
                 test_loader: DataLoader, attacker, params: Dict,
                 checkpoint_path: str = None):
        super(RobustPlusRegularizationTrainer, self).__init__(model, train_loader, test_loader,
                                                              attacker, params, checkpoint_path)
        self._blocks = WRN34Block(model)
        self._register_forward_hook_to_k_block(k)
        self._hooked_features_list = []
        self._lambda = _lambda

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
        inputs, labels = inputs.to(self._device), labels.to(self._device)

        adv_inputs = self._gen_adv(inputs, labels)
        adv_outputs = self.model(adv_inputs)
        clean_outputs = self.model(inputs)

        r_adv = self._hooked_features_list[0]
        r_clean = self._hooked_features_list[1]
        # so stupid!
        regularization_term = self._lambda * torch.norm(
            (r_adv - r_clean).view(r_adv[0], -1),
            dim=1
        ).sum()
        l_term = self.criterion(adv_outputs, labels)
        print(f"regularization: {regularization_term}")
        self._hooked_features_list.clear()

        loss = l_term + regularization_term

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        batch_training_acc = (clean_outputs.argmax(dim=1) == labels).float().mean().item()
        batch_running_loss = loss.item()

        return batch_running_loss, batch_training_acc

    def _register_forward_hook_to_k_block(self, k):
        assert 1 <= k <= 17
        self._first = True
        block = getattr(self._blocks, f"block{18-k}")
        block[-1].register_forward_hook(self.get_layer_outputs)

    def get_layer_outputs(self, layer, inputs, outputs):
        if self._first:
            print(layer)
            self._first = False
        if self.model.training:
            self._hooked_features_list.append(outputs.clone().detach())