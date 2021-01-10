from typing import Dict
import time

import torch
from torch.utils.data import DataLoader

import numpy as np

from ..adv_trainer import ADVTrainer
from ..mixins import InitializeTensorboardMixin
from src.utils import logger
from src.networks import SupportedAllModuleType, make_blocks
from src.utils.spectral_norm import spectral_norm, remove_spectral_norm


class RobustPlusSpectrumNormTrainer(ADVTrainer, InitializeTensorboardMixin):
    model:torch.nn.Module
    
    def __init__(self, model: SupportedAllModuleType, train_loader: DataLoader, test_loader: DataLoader, 
                    attacker, params: Dict, checkpoint_path: str, beta_norm: float=1.0, power_iter:int=1):
        
        # Ugly Hack
        # For adapt 'BaseTrainer', since it loads checkpoint during init before layers add spectral norm
        self.spectral_norm_initialized = False
        super().__init__(model, train_loader, test_loader, attacker, params, checkpoint_path=checkpoint_path)

        self._beta_norm = beta_norm
        self._power_iter = power_iter
        self._apply_spectral_norm()

        self.spectral_norm_initialized = True
        # Here we actually try to load checkpoint.
        if checkpoint_path:
            import os
            self._checkpoint_path = checkpoint_path
            if os.path.exists(checkpoint_path):
                logger.warning("We load checkpoint here")
                self._load_from_checkpoint(checkpoint_path)
        
        self.summary_writer = self.init_writer()

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
        inputs, labels = inputs.to(self._device), labels.to(self._device)

        self._freeze_all_layers()
        adv_inputs = self._gen_adv(inputs, labels)
        self._unfreeze_all_layers()

        adv_outputs = self.model(adv_inputs) #type:torch.Tensor

        loss = self.criterion(adv_outputs, labels) #type:torch.Tensor

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._robust_acc += (adv_outputs.argmax(dim=1) == labels).float().mean().item()
        batch_running_loss = loss.item()

        return batch_running_loss

    def train(self, save_path):
        batch_number = len(self._train_loader)
        best_robustness = self.best_acc
        start_epoch = self.start_epoch

        logger.info(f"starting epoch: {start_epoch}")
        logger.info(f"start lr: {self.current_lr}")
        logger.info(f"best robustness: {best_robustness}")

        for ep in range(start_epoch, self._train_epochs + 1):
            self._adjust_lr(ep)

            # show current learning rate
            logger.debug(f"lr: {self.current_lr}")

            running_loss = 0.0
            # record current robustness
            self._robust_acc = 0

            start_time = time.perf_counter()

            for index, data in enumerate(self._train_loader):
                batch_running_loss = self.step_batch(data[0], data[1])

                running_loss += batch_running_loss

                # warm up learning rate
                if ep <= self._warm_up_epochs:
                    self.warm_up_scheduler.step()

                if index % batch_number == batch_number - 1:
                    end_time = time.perf_counter()

                    acc = self.test()
                    average_train_loss = (running_loss / batch_number)
                    average_robust_accuracy = self._robust_acc / batch_number
                    epoch_cost_time = end_time - start_time

                    # write loss, time, test_acc, train_acc to tensorboard
                    if hasattr(self, "summary_writer"):
                        self.summary_writer.add_scalar("train loss", average_train_loss, ep)
                        self.summary_writer.add_scalar("test accuracy", acc, ep)
                        self.summary_writer.add_scalar("time per epoch", epoch_cost_time, ep)
                        self.summary_writer.add_scalar("best robustness", average_robust_accuracy, ep)

                    logger.info(
                        f"epoch: {ep}   loss: {average_train_loss:.6f}   "
                        f"test accuracy: {acc}   robust accuracy: {average_robust_accuracy}   "
                        f"time: {epoch_cost_time:.2f}s")

                    if best_robustness < average_robust_accuracy:
                        best_robustness = average_robust_accuracy
                        logger.info(f"better robustness: {best_robustness}")
                        logger.info(f"corresponding accuracy on test set: {acc}")
                        self._save_model(f"{save_path}-best_robust")

            self._save_checkpoint(ep, best_robustness)

        logger.info("finished training")
        logger.info(f"best robustness on test set: {best_robustness}")

        self._save_last_model(f"{save_path}-last") # imTyrant added it for saving last model.
    
    # spectral norm stuffs
    def _apply_spectral_norm(self):
        for name, module in list(self.model.named_modules()):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                setattr(self.model, name, spectral_norm(module, n_power_iterations=self._power_iter, norm_beta=self._beta_norm))
                logger.debug(f"replace '{name}'  by SN version, with 'n_power_iterations'={self._power_iter}, 'norm_beta'={self._beta_norm}")
    
    def _remove_spectral_norm(self):
        for name, module in list(self.model.named_modules()):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                setattr(self.model, name, remove_spectral_norm(module))
                logger.debug(f"recover '{name}' to normal version")

    def _unfreeze_all_layers(self):
        for p in self.model.parameters():
            p.requires_grad = True

    # freez model for speedup
    def _freeze_all_layers(self):
        for p in self.model.parameters():
            p.requires_grad = False
    
    # overload checkpointing stuffs
    def _save_checkpoint(self, current_epoch, best_acc):
        return super()._save_checkpoint(current_epoch, best_acc)
    
    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        if not self.spectral_norm_initialized:
            logger.warning("We don't load checkpoint at this moment")
            # here we give some fake data
            self.start_epoch = 1
            self.best_acc = 0
        else:
            super()._load_from_checkpoint(checkpoint_path)

    # overload saving last model
    def _save_last_model(self, save_path: str) -> None:
        self._remove_spectral_norm()
        super()._save_last_model(save_path)
