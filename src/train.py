import sys
import os
from dataclasses import dataclass
from typing import Dict
import time

from utils import WarmUpLR
from models import resnet
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.nn.modules import loss
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader
from tl import TLResNet, TLWideResNet
import json
from random import shuffle
from hyperparameters import *

# multiple gpu
from torch.nn import DataParallel
# from parallize import BalancedDataParallel

# if the batch_size and model structure is fixed, this may accelerate the training process
torch.backends.cudnn.benchmark = True

from attack import PGDAttack



@dataclass
class HyperParameter:
    scheduler: optim.lr_scheduler
    optimizer: Optimizer
    # the loss function
    criterion: loss._Loss
    batch_size: int
    epochs: int
    device: str


class Trainer:

    def __init__(self, model: Module, train_loader, test_loader: DataLoader,
                 device=DEFAULT_DEVICE, lr=DEFAULT_LR, momentum=DEFAULT_MOMENTUM,
                 epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                 parallelism=DEFAULT_PARALLELISM, milestones=MILESTONES,
                 gamma=0.2, warm_phases=WARM_PHASES, criterion=loss.CrossEntropyLoss(),
                 checkpoint_path=None):
        print("initialize trainer")

        # parameter pre-processing
        self.test_loader = test_loader
        self.train_loader = train_loader

        model.to(device)
        self.parallelism = parallelism
        if torch.cuda.device_count() > 1 and parallelism:
            print(f"using {torch.cuda.device_count()} GPUs")
            self.model = DataParallel(model)
        # self.model = BalancedDataParallel(128, model)
        else:
            self.model = model

        optimizer = optim.SGD(
            # choose whether train or not
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            momentum=momentum,
            weight_decay=5e-4
        )

        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        start_epoch = 1
        # best accuracy of current model
        best_acc = 0

        if checkpoint_path:
          checkpoint = self._load_from_checkpoint(checkpoint_path)
          self.model.load_state_dict(checkpoint.get("parameters"))
          optimizer.load_state_dict(checkpoint.get("optimizer"))
          train_scheduler.load_state_dict(checkpoint.get("lr_scheduler"))
          start_epoch = checkpoint.get("last_epoch") + 1
          best_acc = checkpoint.get("best_acc")

        self.start_epoch = start_epoch
        self.best_acc = best_acc


        # warm phases
        self.warm_phases = warm_phases
        # warmup learning rate
        self.warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * self.warm_phases)

        self.hp = HyperParameter(
            scheduler=train_scheduler,
            optimizer=optimizer,
            criterion=criterion,
            batch_size=batch_size,
            epochs=epochs, 
            device=device
        )

        # self.train_loader = train_loader
        print("initialize finished")
        print(f"hyper parameter: {self.hp}")

    def train(self, save_path, attack=False, attacker=None, params: Dict = None):
        self._init_attacker(attack, attacker, params)

        batch_number = len(self.train_loader)

        
        best_acc = self.best_acc
        start_epoch = self.start_epoch



        print(f"starting epoch: {start_epoch}")
        print(f"start lr: {self.hp.optimizer.param_groups[0].get('lr')}")
        print(f"best accuracy: {best_acc}")

        for ep in range(start_epoch, self.hp.epochs + 1):
            
            if ep > self.warm_phases:
                self._adjust_lr(ep)

            # show current learning rate
            print(f"lr: {self.hp.optimizer.param_groups[0].get('lr')}")
            
            training_acc, running_loss = 0, .0
            start_time = time.process_time()

            for index, data in enumerate(self.train_loader):

                # print(f"lr: {self.hp.scheduler.get_last_lr()}")
                inputs, labels = data[0].to(self.hp.device), data[1].to(self.hp.device)

                self.hp.optimizer.zero_grad()
                if attack:
                    # calculate this first, for this will zero the grad
                    adv_inputs = self.attacker.calc_perturbation(inputs, labels)
                    # zero the grad
                    self.hp.optimizer.zero_grad()
                    # feature extractor
                    z_outputs = self.model.feature_extractor(inputs)
                    z_adv_outputs = self.model.feature_extractor(adv_inputs)
                    # adv_outputs = self.model(adv_inputs)
                    # outputs = self.model(inputs)
                    adv_outputs = self.model.fc(z_adv_outputs.view(z_adv_outputs.size(0), -1))
                    outputs = self.model.fc(z_outputs.view(z_outputs.size(0), -1))

                    _loss = 0.005 * torch.norm(z_outputs - z_adv_outputs).item() + 1.0 * self.hp.criterion(outputs, labels) + self.hp.criterion(adv_outputs, labels)
                    # _loss = self.hp.criterion(adv_outputs, labels)
                else:
                    outputs = self.model(inputs)

                    # outputs_before_fc = self.model.feature_extractor(inputs)
                    _loss = self.hp.criterion(outputs, labels)
                    # _loss += + 0.005 * torch.norm(outputs_before_fc - pre_output).item()

                _loss.backward()
                self.hp.optimizer.step()
                # torch.cuda.empty_cache()

                outputs: torch.Tensor
                # adv_outputs: torch.Tensor
                # todo
                training_acc += (outputs.argmax(dim=1) == labels).float().mean().item()

                # warm up learning rate
                if ep <= self.warm_phases:
                    self.warmup_scheduler.step()

                running_loss += _loss.item()

                if index % batch_number == batch_number - 1:
                    end_time = time.process_time()

                    acc = self.test(self.model, test_loader=self.test_loader, device=self.hp.device)
                    print(
                        f"epoch: {ep}   loss: {(running_loss / batch_number):.6f}   train accuracy: {training_acc / batch_number}   "
                        f"test accuracy: {acc}   time: {end_time - start_time:.2f}s")

                    if best_acc < acc:
                        best_acc = acc
                        self._save_best_model(save_path, ep, acc)

            # fixme 
            # this is not work
            # this question is link to https://github.com/pytorch/pytorch/pull/26423
            self.hp.scheduler.step()
            self._save_checkpoint(save_path, ep, acc)

        # if self.parallelism:
        #     torch.save(self.model.module.state_dict(), f"{save_path}-latest")
        # else:
        #     torch.save(self.model.state_dict(), f"{save_path}-latest")
        print("finished training")
        print(f"best accuracy on test set: {best_acc}")

    @staticmethod
    def test(model: Module, test_loader, device, debug=False):
        model.eval()
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                _, y_hats = model(inputs).max(1)
                match = (y_hats == labels)
                correct += len(match.nonzero())

        if debug:
            print(f"Testing: {len(test_loader.dataset)}")
            print(f"correct: {correct}")
            print(f"accuracy: {100 * correct / len(test_loader.dataset):.3f}%")

        model.train()

        return correct / len(test_loader.dataset)

    def _init_attacker(self, attack, attacker, params):
        self.attack = attack
        if attack:
            print(f"robustness training with {attacker.__name__}")
            self.attacker = attacker(self.model, **params)
            self.attacker.print_params()
        else:
            print("normal training")

    def _save_best_model(self, save_path, current_epochs, accuracy):
        """save best model with current info"""
        info = {
            "current_epochs": current_epochs,
            "total_epochs": self.hp.epochs,
            "accuracy": accuracy
        }
        if self.attack:
            info.update({
                "attack": self.attack,
                "attacker": type(self.attacker).__name__,
                "epsilons": self.attacker.epsilon,
            })
        with open(os.path.join(os.path.dirname(save_path), "info.json"), "w", encoding="utf8") as f:
            json.dump(info, f)
        # multiple gpus
        if self.parallelism:
            torch.save(self.model.module.state_dict(), f"{save_path}-best")
        else:
            torch.save(self.model.state_dict(), f"{save_path}-best")

    def _adjust_lr(self, ep):
        if ep <= 40:
            lr = 0.1
        elif ep <= 70:
            lr = 0.02
        elif ep <= 90:
            lr = 0.004
        else:
            lr = 0.0008

        for param_group in self.hp.optimizer.param_groups:
            param_group['lr'] = lr

    def _save_checkpoint(self, save_path, last_epoch, best_acc):
      if self.parallelism:
        parameters = self.model.module.state_dict()
      else:
        parameters = self.model.state_dict()

      optimizer = self.hp.optimizer.state_dict()
      lr_scheduler = self.hp.scheduler.state_dict()

      torch.save({
          "parameters": parameters,
          "optimizer": optimizer,
          "lr_scheduler": lr_scheduler,
          "last_epoch": last_epoch,
          "best_acc": best_acc 
      }, f"{os.path.join(os.path.dirname(save_path), 'checkpoint.pth')}")

    def _load_from_checkpoint(self, checkpoint_path):
      return torch.load(checkpoint_path)

    @staticmethod
    def train_tl(origin_model_path, save_path, train_loader,
                 test_loader, device, choice="wrn34_10", num_classes=10, droprate=0):
        print(f"transform learning on model: {origin_model_path}")
        model = TLWideResNet.create_model(choice, droprate=droprate, num_classes=num_classes)
        model.load_model(origin_model_path)
        trainer = Trainer(model=model, train_loader=train_loader, test_loader=test_loader, device=device)
        trainer.train(save_path)


if __name__ == '__main__':
    pass
    