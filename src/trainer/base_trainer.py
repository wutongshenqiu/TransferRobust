import os
import time
import json
from typing import Tuple

import torch
from torch import optim
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader

from src import settings
from utils import WarmUpLR, evaluate_accuracy


class BaseTrainer:
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

        if checkpoint_path:
            self._checkpoint_path = checkpoint_path
            if os.path.exists(checkpoint_path):
                self._load_from_checkpoint(checkpoint_path)
            else:
                self.start_epoch = 1
                # best accuracy of current model
                self.best_acc = 0

        print("initialize finished")
        print(f"parameters: ")
        self.print_parameters()

    def train(self, save_path):
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
                batch_running_loss, batch_training_acc = self.step_batch(data[0], data[1])

                training_acc += batch_training_acc
                running_loss += batch_running_loss

                # warm up learning rate
                if ep <= self._warm_up_epochs:
                    self.warm_up_scheduler.step()

                if index % batch_number == batch_number - 1:
                    end_time = time.process_time()

                    acc = self.test()
                    print(
                        f"epoch: {ep}   loss: {(running_loss / batch_number):.6f}   train accuracy: {training_acc / batch_number}   "
                        f"test accuracy: {acc}   time: {end_time - start_time:.2f}s")

                    if best_acc < acc:
                        best_acc = acc
                        self._save_best_model(save_path, ep, acc)

            self._save_checkpoint(ep, best_acc)

        print("finished training")
        print(f"best accuracy on test set: {best_acc}")

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        raise NotImplementedError("must overwrite method `step_epoch`")

    def test(self):
        return evaluate_accuracy(self.model, self._test_loader, self._device)

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
            "best_accuracy": accuracy
        }
        with open(os.path.join(os.path.dirname(save_path), "info.json"), "w", encoding="utf8") as f:
            json.dump(info, f)
        torch.save(self.model.state_dict(), f"{save_path}-best")

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

    def _save_checkpoint(self, current_epoch, best_acc):
        model_weights = self.model.state_dict()
        optimizer = self.optimizer.state_dict()

        torch.save({
            "model_weights": model_weights,
            "optimizer": optimizer,
            "current_epoch": current_epoch,
            "best_acc": best_acc
        }, f"{self._checkpoint_path}")

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint.get("model_weights"))
        self.optimizer.load_state_dict(checkpoint.get("optimizer"))
        start_epoch = checkpoint.get("current_epoch") + 1
        best_acc = checkpoint.get("best_acc")

        self.start_epoch = start_epoch
        self.best_acc = best_acc