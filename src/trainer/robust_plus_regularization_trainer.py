from typing import Dict

import torch
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from trainer import ADVTrainer, WRN34Block
from utils import logger


class RobustPlusRegularizationTrainer(ADVTrainer):
    def __init__(self, k: int, _lambda: float, model: Module, train_loader: DataLoader,
                 test_loader: DataLoader, attacker, params: Dict,
                 checkpoint_path: str = None, log_dir: str = None):
        super(RobustPlusRegularizationTrainer, self).__init__(model, train_loader, test_loader,
                                                              attacker, params, checkpoint_path)
        self._blocks = WRN34Block(model)
        self._register_forward_hook_to_k_block(k)
        self._hooked_features_list = []
        self._lambda = _lambda

        # tensorboard
        self.writer = SummaryWriter(log_dir=log_dir)
        if not hasattr(self, "_current_batch"):
            self._current_batch = 0

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
        logger.debug(f"d_loss: {regularization_term}")

        self._hooked_features_list.clear()

        l_term = self.criterion(adv_outputs, labels)
        logger.debug(f"l_loss: {regularization_term}")

        # tensorboard draw
        self.writer.add_scalars(
            f"lambda_{self._lambda}",
            {
                "l_loss": l_term,
                "d_loss": regularization_term
            }
        )
        self.writer.add_scalar(
            f"lambda_{self._lambda}_l_loss",
            l_term,
            self._current_batch
        )
        self.writer.add_scalar(
            f"lambda_{self._lambda}_d_loss",
            regularization_term,
            self._current_batch
        )
        self._current_batch += 1

        loss = l_term + regularization_term

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        batch_training_acc = (clean_outputs.argmax(dim=1) == labels).float().mean().item()
        batch_running_loss = loss.item()

        return batch_running_loss, batch_training_acc

    def _register_forward_hook_to_k_block(self, k):
        assert 1 <= k <= 17
        block = getattr(self._blocks, f"block{18 - k}")
        block[-1].register_forward_hook(self.get_layer_outputs)

    def get_layer_outputs(self, layer, inputs, outputs):
        if self.model.training:
            self._hooked_features_list.append(outputs.clone().detach())

    def _save_checkpoint(self, current_epoch, best_acc):
        model_weights = self.model.state_dict()
        optimizer = self.optimizer.state_dict()
        current_batch = self._current_batch

        torch.save({
            "model_weights": model_weights,
            "optimizer": optimizer,
            "current_epoch": current_epoch,
            "best_acc": best_acc,
            "current_batch": current_batch
        }, f"{self._checkpoint_path}")

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint.get("model_weights"))
        self.optimizer.load_state_dict(checkpoint.get("optimizer"))
        start_epoch = checkpoint.get("current_epoch") + 1
        best_acc = checkpoint.get("best_acc")

        self.start_epoch = start_epoch
        self.best_acc = best_acc

        self._current_batch = checkpoint.get("current_batch")
