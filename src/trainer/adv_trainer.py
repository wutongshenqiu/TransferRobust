from typing import Dict
import warnings

import torch
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utils import init_attacker, init_classifier, logger
from . import BaseTrainer


class BaseADVTrainer(BaseTrainer):
    def __init__(self, model: Module, train_loader: DataLoader,
                 test_loader: DataLoader, attacker, params: Dict,
                 checkpoint_path: str = None):
        super(BaseADVTrainer, self).__init__(model, train_loader, test_loader, checkpoint_path)
        self.attacker = self._init_attacker(attacker, params)

    def _init_attacker(self, attacker, params):
        raise NotImplementedError("must overwrite method `init_attacker`")


class ADVTrainer(BaseADVTrainer):
    def __init__(self, model: Module, train_loader: DataLoader,
                 test_loader: DataLoader, attacker, params: Dict,
                 checkpoint_path: str = None):
        super(ADVTrainer, self).__init__(model, train_loader, test_loader, attacker, params, checkpoint_path)

    def _init_attacker(self, attacker, params):
        attacker = attacker(self.model, **params)
        attacker.print_parameters()

        return attacker

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
        inputs, labels = inputs.to(self._device), labels.to(self._device)

        adv_inputs = self._gen_adv(inputs, labels)
        outputs = self.model(adv_inputs)
        # outputs = self.model(inputs)
        # loss = self.criterion(outputs, labels) + self.criterion(adv_outputs, labels)
        loss = self.criterion(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        batch_training_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        batch_running_loss = loss.item()

        return batch_running_loss, batch_training_acc

    def _gen_adv(self, inputs: torch.Tensor, labels: torch.Tensor):
        self.model.eval()

        adv_inputs = self.attacker.calc_perturbation(inputs, labels)
        adv_inputs = adv_inputs.to(self._device)

        # self.optimizer.zero_grad()
        self.model.train()

        return adv_inputs


# fixme
class ARTTrainer(BaseADVTrainer):
    def __init__(self, model: Module, train_loader: DataLoader,
                 test_loader: DataLoader, attacker: str, params: Dict,
                 dataset_mean, dataset_std, checkpoint_path: str = None):
        warnings.warn("can not work well now!")
        super(ARTTrainer, self).__init__(model, train_loader, test_loader, attacker, params, checkpoint_path)
        self._init_normalize(dataset_mean, dataset_std)

    def _init_attacker(self, attacker, params):
        logger.info(f"robustness training with {attacker}")
        classifier = init_classifier(model=self.model)
        return init_attacker(classifier, attacker, params)

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
        inputs, labels = inputs.to(self._device), labels.to(self._device)

        adv_inputs = self._gen_adv(inputs)
        adv_outputs = self.model(adv_inputs)
        # self._apply_normalize(inputs)
        # outputs = self.model(inputs)
        # loss = self.criterion(outputs, labels) + self.criterion(adv_outputs, labels)
        # loss.backward()
        loss = self.criterion(adv_outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        batch_training_acc = (adv_outputs.argmax(dim=1) == labels).float().mean().item()
        batch_running_loss = loss.item()

        return batch_running_loss, batch_training_acc

    def _gen_adv(self, inputs: torch.Tensor):
        self.model.eval()

        adv_inputs = torch.from_numpy(self.attacker.generate(x=inputs.cpu().numpy()))
        adv_inputs = adv_inputs.to(self._device)

        # self.optimizer.zero_grad()
        self.model.train()

        return adv_inputs

    def _init_normalize(self, mean, std):
        self._normalize = transforms.Normalize(mean=mean, std=std, inplace=True)

    def _apply_normalize(self, batch_tensor: torch.Tensor):
        with torch.no_grad():
            for t in batch_tensor[:]:
                self._normalize(t)
