import os
import time
import json
from typing import Dict, Tuple, Callable, Any

import torch
from torch import optim
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader
from torchvision import transforms

from networks.resnet import resnet18
from networks.wrn import wrn34_10
from tl import TLWideResNet
from config import settings
from utils import WarmUpLR, evaluate_accuracy
from art_utils import init_attacker, init_classifier

# if the batch_size and model structure is fixed, this may accelerate the training process
torch.backends.cudnn.benchmark = True


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

        if checkpoint_path and os.path.exists(checkpoint_path):
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
                sub_running_loss, sub_training_acc = self.step_batch(data[0], data[1])

                training_acc += sub_training_acc
                running_loss += sub_running_loss

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

            self._save_checkpoint(save_path, ep, best_acc)

        print("finished training")
        print(f"best accuracy on test set: {best_acc}")

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
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
                 test_loader, choice="wrn34_10", num_classes=10, droprate=0):
        print(f"transform learning on model: {origin_model_path}")
        model = TLWideResNet.create_model(choice, droprate=droprate, num_classes=num_classes)
        model.load_model(origin_model_path)
        trainer = BaseTrainer(model=model, train_loader=train_loader, test_loader=test_loader)
        trainer.train(save_path)


class NormalTrainer(BaseTrainer):
    def __init__(self, model: Module, train_loader: DataLoader,
                 test_loader: DataLoader, checkpoint_path: str = None):
        super(NormalTrainer, self).__init__(model, train_loader, test_loader, checkpoint_path)

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        inputs, labels = inputs.to(self._device), labels.to(self._device)
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        sub_training_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        sub_running_loss = loss.item()

        return sub_running_loss, sub_training_acc


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
        print(f"robustness training with {type(attacker).__name__}")
        attacker = attacker(self.model, **params)
        attacker.print_parameters()

        return attacker

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
        inputs, labels = inputs.to(self._device), labels.to(self._device)
        self.optimizer.zero_grad()

        adv_inputs = self._gen_adv(inputs, labels)
        adv_outputs = self.model(adv_inputs)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels) + self.criterion(adv_outputs, labels)
        loss.backward()
        self.optimizer.step()

        sub_training_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        sub_running_loss = loss.item()

        return sub_running_loss, sub_training_acc

    def _gen_adv(self, inputs: torch.Tensor, labels: torch.Tensor):
        self.model.eval()

        adv_inputs = self.attacker.generate(inputs, labels)
        adv_inputs = adv_inputs.to(self._device)

        self.optimizer.zero_grad()
        self.model.train()

        return adv_inputs


class ARTTrainer(BaseADVTrainer):
    def __init__(self, model: Module, train_loader: DataLoader,
                 test_loader: DataLoader, attacker: str, params: Dict,
                 dataset_mean, dataset_std, checkpoint_path: str = None):
        super(ARTTrainer, self).__init__(model, train_loader, test_loader, attacker, params, checkpoint_path)
        self._init_normalize(dataset_mean, dataset_std)

    def _init_attacker(self, attacker, params):
        print(f"robustness training with {attacker}")
        classifier = init_classifier(model=self.model)
        return init_attacker(classifier, attacker, params)

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
        inputs, labels = inputs.to(self._device), labels.to(self._device)
        self.optimizer.zero_grad()

        adv_inputs = self._gen_adv(inputs)
        adv_outputs = self.model(adv_inputs)
        self._apply_normalize(inputs)
        outputs = self.model(inputs)

        loss = self.criterion(outputs, labels) + self.criterion(adv_outputs, labels)
        loss.backward()
        self.optimizer.step()

        sub_training_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        sub_running_loss = loss.item()

        return sub_running_loss, sub_training_acc

    def _gen_adv(self, inputs: torch.Tensor):
        self.model.eval()

        adv_inputs = torch.from_numpy(self.attacker.generate(x=inputs.cpu().numpy()))
        adv_inputs = adv_inputs.to(self._device)

        self.optimizer.zero_grad()
        self.model.train()

        return adv_inputs

    def _init_normalize(self, mean, std):
        self._normalize = transforms.Normalize(mean=mean, std=std, inplace=True)

    def _apply_normalize(self, batch_tensor: torch.Tensor):
        with torch.no_grad():
            for t in batch_tensor[:]:
                self._normalize(t)


if __name__ == '__main__':
    from utils import get_cifar_testing_dataloader, get_cifar_training_dataloader, CIFAR100_TRAIN_STD, CIFAR100_TRAIN_MEAN
    from art_utils import attack_params

    model = resnet18(num_classes=100)
    trainer = ARTTrainer(
        model, get_cifar_training_dataloader("cifar100"),
        get_cifar_testing_dataloader("cifar100"),
        attacker="ProjectedGradientDescent",
        params=attack_params.get("ProjectedGradientDescent"),
        dataset_mean=CIFAR100_TRAIN_MEAN,
        dataset_std=CIFAR100_TRAIN_STD,
        checkpoint_path="./checkpoint.pth"
    )

    trainer.train("./test_resnet")