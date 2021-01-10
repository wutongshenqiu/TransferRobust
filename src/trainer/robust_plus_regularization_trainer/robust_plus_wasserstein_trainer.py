from typing import Dict, Tuple, overload
import time

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader

import numpy as np

from ..adv_trainer import ADVTrainer
from ..mixins import InitializeTensorboardMixin
from src import settings
from src.utils import logger
from src.networks import SupportedAllModuleType, make_blocks
from src.utils.spectral_norm import spectral_norm, remove_spectral_norm

class _Estimator(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.fc1 = spectral_norm(nn.Linear(input_dim, 512))
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = spectral_norm(nn.Linear(512, 256))
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = spectral_norm(nn.Linear(256, 1))
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

class RobustPlusWassersteinTrainer(ADVTrainer, InitializeTensorboardMixin):
    model:SupportedAllModuleType
    _optimE:torch.optim.Optimizer
    _estimator:_Estimator

    def __init__(self, k: int, model: SupportedAllModuleType, train_loader: DataLoader,
                 test_loader: DataLoader, attacker, params: Dict, checkpoint_path: str = None, lr_estimator:float=0.0001, lambda_: float=1.0, n_critic:int=1):

        self._is_initialized = False

        super().__init__(model, train_loader, test_loader, attacker, params, checkpoint_path)

        self._lambda = lambda_
        self._k = k
        self.n_critic = n_critic

        self._blocks = make_blocks(model)

        # we use a dict to store intermediate features since DataParallel cannot guarantee 
        # synchronization for features.
        self._features:Dict[int, torch.Tensor] = {}

        self._register_forward_hook_to_k_block(k)

        self._prepare_estimator(lr_estimator)

        self.summary_writer = self.init_writer()

        if torch.cuda.device_count() <= 1:
            self._is_parallelism = False
            logger.warning("only one gpu is detected, CUDA may be out of memory!")
        else:
            self.model = nn.DataParallel(self.model)
            self._estimator = nn.DataParallel(self._estimator)
            self._is_parallelism = True

        self._is_initialized = True
        if checkpoint_path:
            import os
            self._checkpoint_path = checkpoint_path
            if os.path.exists(checkpoint_path):
                logger.warning("We load checkpoint here")
                self._load_from_checkpoint(checkpoint_path)



    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor, iter:int):
        inputs = inputs.to(self._device, non_blocking=True)
        labels = labels.to(self._device)

        self._freeze_all_layers()
        adv_inputs = self._gen_adv(inputs, labels)
        self._unfreeze_all_layers()

        # cat for speedup
        batch_size = inputs.shape[0]
        big_batch = torch.cat([adv_inputs, inputs], dim=0)
        with torch.no_grad():
            self.model(big_batch)

        ## update estimator ##
        # gather all intermediate features to first card (settings.device)
        features = torch.cat([self._features[i].to(settings.device) for i in range(torch.cuda.device_count())], dim=0)

        # Wasserstein Distance Estimation
        we = self._estimator(features)
        adv_we, clean_we = we[:batch_size], we[batch_size:]
        
        # Wasserstein Distance
        loss_E = - (torch.mean(clean_we) - torch.mean(adv_we)) * self._lambda
        # JS Divergence
        # loss_E = torch.mean(torch.log(we[batch_size:]) - torch.log(1 - we[:batch_size]))

        self._optimE.zero_grad()
        loss_E.backward()
        self._optimE.step()

        """update model"""
        logits = self.model(big_batch)
        adv_logits = logits[:batch_size]
        features = torch.cat([self._features[i].to(settings.device).detach() for i in range(torch.cuda.device_count())], dim=0)
        critic = self._estimator(features)
        adv_critic, clean_critic = critic[:batch_size], critic[batch_size:]

        # Wasserstein Distance
        loss_C =  (torch.mean(clean_critic) - torch.mean(adv_critic)) * self._lambda
        loss_M = self.criterion(adv_logits, labels) + loss_C #type:torch.Tensor
        # JS Divergence
        # loss_M = self.criterion(logits, labels) - torch.mean(torch.log(critic)) #type:torch.Tensor

        self.optimizer.zero_grad()
        loss_M.backward()
        self.optimizer.step()

        batch_robust_acc = adv_logits.argmax(dim=1).eq(labels).sum().item()
        self._robust_acc += batch_robust_acc

            # print("E loss", loss_E, "M loss", loss_M, "critic", torch.mean(critic), "acc", batch_robust_acc)

        self._features.clear()

        return loss_M.item(), loss_E.item(), loss_C.item(),  batch_robust_acc
    
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

            training_acc, running_loss = 0, .0
            estimator_loss, critic_loss = 0., 0.
            # record current robustness
            self._robust_acc = 0

            start_time = time.perf_counter()

            for index, data in enumerate(self._train_loader):
                batch_running_loss, batch_estimator_loss, batch_critic_loss, batch_training_acc = self.step_batch(data[0], data[1], index)

                training_acc += batch_training_acc
                running_loss += batch_running_loss
                estimator_loss += batch_estimator_loss
                critic_loss += batch_critic_loss

                # warm up learning rate
                if ep <= self._warm_up_epochs:
                    self.warm_up_scheduler.step()

                if index % batch_number == batch_number - 1:
                    end_time = time.perf_counter()

                    acc = self.test()
                    average_train_loss = (running_loss / batch_number)
                    average_train_accuracy = training_acc / batch_number
                    average_robust_accuracy = self._robust_acc / batch_number
                    average_estimator_loss = estimator_loss / batch_number
                    average_critic_loss = critic_loss / batch_number
                    epoch_cost_time = end_time - start_time

                    # write loss, time, test_acc, train_acc to tensorboard
                    if hasattr(self, "summary_writer"):
                        self.summary_writer.add_scalar("train loss", average_train_loss, ep)
                        self.summary_writer.add_scalar("estimator loss", average_estimator_loss, ep)
                        self.summary_writer.add_scalar("critic loss", average_critic_loss, ep)
                        self.summary_writer.add_scalar("train accuracy", average_train_accuracy, ep)
                        self.summary_writer.add_scalar("test accuracy", acc, ep)
                        self.summary_writer.add_scalar("time per epoch", epoch_cost_time, ep)
                        self.summary_writer.add_scalar("best robustness", average_robust_accuracy, ep)

                    logger.info(
                        f"epoch: {ep}   loss: {average_train_loss:.6f}  estimator loss {average_estimator_loss:.6f}   critic loss {average_critic_loss:.6f}   "
                        f"train accuracy: {average_train_accuracy}   "
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

        self._remove_hook() # remove the hook

    def _prepare_estimator(self, lr_estimator:float)->_Estimator:
        # first, we get an input for calculating dimension of features
        image, _ = next(iter(self._train_loader))
        image = image.to(self._device)

        with torch.no_grad():
            self.model(image)
        
        assert 0 in self._features
        dim = torch.prod(torch.tensor(self._features[0].shape[1:]))

        self._estimator = _Estimator(dim).to(self._device)
        self._optimE = torch.optim.RMSprop(self._estimator.parameters(), lr=lr_estimator)

        logger.debug(f"Wasserstein estimator is initialized, lr={lr_estimator}.")

    

    def _register_forward_hook_to_k_block(self, k):
        def _get_layer_inputs(layer, inputs:Tuple[torch.Tensor], outputs:torch.Tensor):
            if self.model.training:
                self._features[inputs[0].device.index] = inputs[0].clone()
            return outputs

        total_blocks = self._blocks.get_total_blocks()
        assert 1 <= k <= total_blocks
        logger.debug(f"model total blocks: {total_blocks}")
        logger.debug(f"register hook to the fist layer of {k}th block from last")
        block = getattr(self._blocks, f"block{total_blocks-k+1}") # type:torch.nn.Module
        
        self._hook_handle = block.register_forward_hook(_get_layer_inputs)
        logger.debug("hook is applied")
    
    def _remove_hook(self):
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()
        logger.debug("hook is removed")

    # freeze model parameters for speedup
    def _unfreeze_all_layers(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def _freeze_all_layers(self):
        for p in self.model.parameters():
            p.requires_grad = False
    
    # stuffs for checkpoint
    def _save_checkpoint(self, current_epoch, best_acc):
        if self._is_parallelism:
            model_weights = self.model.module.state_dict()
            estimator_weight = self._estimator.module.state_dict()
        else:
            model_weights = self.model.state_dict()
            estimator_weight = self._estimator.state_dict()
        optimizer = self.optimizer.state_dict()

        torch.save({
            "model_weights": model_weights,
            "optimizer": optimizer,
            "current_epoch": current_epoch,
            "best_acc": best_acc,
            "estimator_weights": estimator_weight
        }, f"{self._checkpoint_path}")

        # Added by imTyrant
        # For saving 'numpy' and 'torch' random state.
        if hasattr(settings, "save_rand_state") and settings.save_rand_state:
            from src.utils import RandStateSnapshooter
            RandStateSnapshooter.lazy_take(f"{self._checkpoint_path}.rand")
            logger.debug(f"random state is saved to '{self._checkpoint_path}.rand'")
    
    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        if not self._is_initialized:
            logger.warning("We don't load checkpoint at this moment")
            # here we give some fake data
            self.start_epoch = 1
            self.best_acc = 0
        else:
            logger.warning("trainer that needed reset blocks may not support load from checkpoint!")
            checkpoint = torch.load(checkpoint_path)
            self.optimizer.load_state_dict(checkpoint.get("optimizer"))
            start_epoch = checkpoint.get("current_epoch") + 1
            best_acc = checkpoint.get("best_acc")

            if self._is_parallelism:
                self.model.module.load_state_dict(checkpoint.get("model_weights"))
                self._estimator.module.load_state_dict(checkpoint.get("estimator_weights"))
            else:
                self.model.load_state_dict(checkpoint.get("model_weights"))
                self._estimator.load_state_dict(checkpoint.get("estimator"))

            self.start_epoch = start_epoch
            self.best_acc = best_acc

            # Added by imTyrant
            # For loading and setting random state.
            if hasattr(settings, "save_rand_state") and settings.save_rand_state:
                from src.utils import RandStateSnapshooter
                import os
                
                if not os.path.exists(f"{self._checkpoint_path}.rand"):
                    return

                RandStateSnapshooter.lazy_set(f"{self._checkpoint_path}.rand")
                logger.warning(f"loaded random state from '{self._checkpoint_path}.rand'")


    def _save_model(self, save_path: str):
        if self._is_parallelism:
            torch.save(self.model.module.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)