import os
import time
import json
from typing import Dict

import torch
from torch import optim
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader

from networks.resnet import resnet18
from tl import TLWideResNet
from config import settings
from utils import WarmUpLR


# if the batch_size and model structure is fixed, this may accelerate the training process
torch.backends.cudnn.benchmark = True


class Trainer:

    def __init__(self, model: Module, train_loader: DataLoader,
                 test_loader: DataLoader, checkpoint_path: str = None):
        print("initialize trainer")
        # can not change the order
        self._init_hyperparameters()
        self._init_model(model)
        self._init_dataloader(train_loader, test_loader)
        self._init_optimizer()
        self._init_scheduler()
        self._init_criterion()

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_from_checkpoint(checkpoint_path)
        else:
            self.start_epoch = 1
            # best accuracy of current model
            self.best_acc = 0

        print("initialize finished")
        print(f"parameters: ")
        self.print_parameters()

    def train(self, save_path, attack=False, attacker=None, params: Dict = None):
        self._init_attacker(attack, attacker, params)

        batch_number = len(self._train_loader)
        best_acc = self.best_acc
        start_epoch = self.start_epoch

        print(f"starting epoch: {start_epoch}")
        print(f"start lr: {self.current_lr}")
        print(f"best accuracy: {best_acc}")

        for ep in range(start_epoch, self._train_epochs + 1):

            self._adjust_lr(ep)

            # show current learning rate
            print(f"lr: {self.current_lr}")

            training_acc, running_loss = 0, .0
            start_time = time.process_time()

            for index, data in enumerate(self._train_loader):

                inputs, labels = data[0].to(self._device), data[1].to(self._device)

                self.optimizer.zero_grad()
                if attack:
                    # calculate this first, for this will zero the grad
                    adv_inputs = self.attacker.calc_perturbation(inputs, labels)
                    # zero the grad
                    self.optimizer.zero_grad()
                    # feature extractor
                    z_outputs = self.model.feature_extractor(inputs)
                    z_adv_outputs = self.model.feature_extractor(adv_inputs)
                    # adv_outputs = self.model(adv_inputs)
                    # outputs = self.model(inputs)
                    adv_outputs = self.model.fc(z_adv_outputs.view(z_adv_outputs.size(0), -1))
                    outputs = self.model.fc(z_outputs.view(z_outputs.size(0), -1))

                    _loss = 0.005 * torch.norm(z_outputs - z_adv_outputs).item() + 1.0 * self.criterion(outputs,
                        labels) + self.criterion(
                        adv_outputs, labels)
                    # _loss = self.hp.criterion(adv_outputs, labels)
                else:
                    outputs = self.model(inputs)

                    # outputs_before_fc = self.model.feature_extractor(inputs)
                    _loss = self.criterion(outputs, labels)
                    # _loss += + 0.005 * torch.norm(outputs_before_fc - pre_output).item()

                _loss.backward()
                self.optimizer.step()
                # torch.cuda.empty_cache()

                outputs: torch.Tensor
                # adv_outputs: torch.Tensor
                # todo
                training_acc += (outputs.argmax(dim=1) == labels).float().mean().item()

                # warm up learning rate
                if ep <= self._warm_up_epochs:
                    self.warm_up_scheduler.step()

                running_loss += _loss.item()

                if index % batch_number == batch_number - 1:
                    end_time = time.process_time()

                    acc = self.test(self.model, test_loader=self._test_loader, device=self._device)
                    print(
                        f"epoch: {ep}   loss: {(running_loss / batch_number):.6f}   train accuracy: {training_acc / batch_number}   "
                        f"test accuracy: {acc}   time: {end_time - start_time:.2f}s")

                    if best_acc < acc:
                        best_acc = acc
                        self._save_best_model(save_path, ep, acc)

            self.optimizer.step()
            self._save_checkpoint(save_path, ep, best_acc)

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

    def _init_dataloader(self, train_loader, test_loader) -> None:
        self._test_loader = test_loader
        self._train_loader = train_loader

    def _init_model(self, model) -> None:
        model.to(self._device)
        self.model = model

    def _init_optimizer(self) -> None:
        self.optimizer = optim.SGD(
            # filter frozen layers
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=settings.start_lr,
            momentum=settings.momentum,
            weight_decay=settings.weight_decay
        )

    def _init_scheduler(self):
        self.warm_up_scheduler = WarmUpLR(self.optimizer, len(self._train_loader) * settings.warm_up_epochs)

    def _init_hyperparameters(self):
        self._batch_size = settings.batch_size
        self._train_epochs = settings.train_epochs
        self._warm_up_epochs = settings.warm_up_epochs
        self._device = torch.device(settings.device if torch.cuda.is_available() else "cpu")

    def _init_criterion(self):
        self.criterion = getattr(torch.nn, settings.criterion)()

    def print_parameters(self) -> None:
        params = {
            "network": type(self.model).__name__,
            "device": str(self._device),
            "train_epochs": str(self._train_epochs),
            "warm_up_epochs": str(self._warm_up_epochs),
            "batch_size": str(self._batch_size),
            "optimizer": str(self.optimizer),
            "criterion": str(self.criterion)
        }

        for key, value in params.items():
            print(f"{key}: {value}")

    def _save_best_model(self, save_path, current_epochs, accuracy):
        """save best model with current info"""
        info = {
            "current_epochs": current_epochs,
            "total_epochs": self._train_epochs,
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

    def _adjust_lr(self, ep):
        if ep > self._warm_up_epochs:
            for step, milestone in enumerate(settings.milestones):
                if ep <= milestone:
                    lr = settings.start_lr * (settings.decrease_rate ** step)
                    break
            else:
                lr = settings.start_lr * (settings.decrease_rate ** len(settings.milestones))

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    @property
    def current_lr(self):
        return self.optimizer.param_groups[0].get('lr')

    def _save_checkpoint(self, save_path, current_epoch, best_acc):
        model_weights = self.model.state_dict()
        optimizer = self.optimizer.state_dict()

        torch.save({
            "model_weights": model_weights,
            "optimizer": optimizer,
            "current_epoch": current_epoch,
            "best_acc": best_acc
        }, f"{os.path.join(os.path.dirname(save_path), 'checkpoint.pth')}")

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint.get("model_weights"))
        self.optimizer.load_state_dict(checkpoint.get("optimizer"))
        start_epoch = checkpoint.get("current_epoch") + 1
        best_acc = checkpoint.get("best_acc")

        self.start_epoch = start_epoch
        self.best_acc = best_acc

    @staticmethod
    def train_tl(origin_model_path, save_path, train_loader,
                 test_loader, device, choice="wrn34_10", num_classes=10, droprate=0):
        print(f"transform learning on model: {origin_model_path}")
        model = TLWideResNet.create_model(choice, droprate=droprate, num_classes=num_classes)
        model.load_model(origin_model_path)
        trainer = Trainer(model=model, train_loader=train_loader, test_loader=test_loader)
        trainer.train(save_path)


if __name__ == '__main__':
    from utils import get_cifar_testing_dataloader, get_cifar_training_dataloader
    model = resnet18(num_classes=100)
    trainer = Trainer(model, get_cifar_training_dataloader("cifar100"),
                      get_cifar_testing_dataloader("cifar100"))

    trainer.train("./test_resnet")