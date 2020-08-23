"""`learning without forgetting` in transfer learning

training steps:
    1. PGD-7 adversarial training on teacher model(e.g. cifar100)
    2. initialize student model from robust teacher model(with reshaped fc layer)
    3. calculate feature representations of **student dataset**(e.g. cifar10 dataset) with initialized student model
    4. store feature representations in memory
        - custom defined Dataloader could be used
    5. use loss: f(x, y_hat) +
                 λ * torch.mean(torch.norm(stored feature representations - running feature representations, p=1, dim=1))
       to train student model with benign student dataset(e.g. cifar10 dataset)
        - in warm-start step only train fully connect(last) layer
        - after warm-start step, train whole model

questions:
    1. `weight decay`: 论文代码中采用 0.0002, 我们一直用的是 0.0005
    2. `epoch`: 论文代码训练了 20000 个 steps, 相当于 51.15 个 epochs, 我们在之前的训练中一直采用 100 个 epochs
    3. `learning rate`: 论文代码采用 0.001 从头至尾, 我们之前的训练中使用初始 lr = 0.1, learning rate decay = 0.2,
                        momentum = [40, 70, 90], 并且我们在第一个 epoch 使用了 warm-up
    4. `warm-start`: 论文代码前 10000 个 steps 只训练 fully connect layer, 之后训练整个模型，我们应该如何设置
"""


from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from typing import Tuple
import time

from src.networks import SupportedModuleType
from .mixins import ReshapeTeacherFCLayerMixin
from ..mixins import InitializeTensorboardMixin
from ..retrain_trainer import ResetBlockMixin, FreezeModelMixin, WRN34Block
from ..base_trainer import BaseTrainer
from src.utils import logger


class LWFTransferLearningTrainer(BaseTrainer, ReshapeTeacherFCLayerMixin,
                                 ResetBlockMixin, FreezeModelMixin, InitializeTensorboardMixin):

    def __init__(self, _lambda: float, teacher_model_path: str,
                 model: SupportedModuleType, train_loader: DataLoader,
                 test_loader: DataLoader, checkpoint_path: str = None):
        """`learning without forgetting` in transfer learning

        Args:
            _lambda: feature representation similarity penalty
        """
        super().__init__(model, train_loader, test_loader, checkpoint_path)

        teacher_state_dict = torch.load(teacher_model_path, map_location=self._device)
        self.reshape_teacher_fc_layer(teacher_state_dict)
        logger.info(f"load from teacher model: \n {teacher_model_path}")
        self.model.load_state_dict(teacher_state_dict)
        # reset fc layer
        self.reset_last_k_blocks(1)

        dataset_with_rft = DatasetWithRobustFeatureRepresentations(train_loader, self.model, self._device)
        self._train_loader = DataLoader(dataset_with_rft, batch_size=train_loader.batch_size,
                                        num_workers=train_loader.num_workers, shuffle=True)
        self._lambda = _lambda

        self.summary_writer = self.init_writer()

    def step_batch(self, inputs: Tuple[Tensor, Tensor], labels: torch.Tensor, optimizer: optim.optimizer) -> Tuple[float, float]:
        inputs, robust_feature_representations = inputs[0].to(self._device), inputs[1].to(self._device)
        labels = labels.to(self._device)

        optimizer.zero_grad()

        outputs = self.model(inputs)
        running_feature_representations = self.model.get_feature_representations()

        loss = self.criterion(outputs, labels) + self._lambda * \
               torch.mean(torch.norm(robust_feature_representations - running_feature_representations, p=1, dim=1))

        loss.backward()
        optimizer.step()

        batch_training_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        batch_running_loss = loss.item()

        return batch_running_loss, batch_training_acc


    def _init_optimizer(self) -> None:
        """override `_init_optimizer` of super class

        we provide two optimizer, `fc_optimizer` and `all_optimizer`
            - `fc_optimizer`: only train fc layer, use for warm-start
            - `all_optimizer`: train all layers
        """

        self.fc_optimizer = optim.SGD(
            self.model.fc.parameters(),
            # fixme
            # lr 如何设置
            lr=NotImplemented,
            momentum=settings.momentum,
            weight_decay=settings.weight_decay
        )

        self.all_optimizer = optim.SGD(
            self.model.parameters(),
            # fixme
            # lr 如何设置
            lr=NotImplemented,
            momentum=settings.momentum,
            weight_decay=settings.weight_decay
        )

    def train(self, save_path):
        """override `train` of super class

        if current epochs < warm-start epochs, we should use optimizer `fc_optimizer`,
        otherwise, we should use optimizer `all_optimizer`
        """
        batch_number = len(self._train_loader)
        best_acc = self.best_acc
        start_epoch = self.start_epoch

        logger.info(f"starting epoch: {start_epoch}")
        logger.info(f"start lr: {self.current_lr}")
        logger.info(f"best accuracy: {best_acc}")

        for ep in range(start_epoch, self._train_epochs + 1):
            # fixme
            # 是否需要调节
            self._adjust_lr(ep)

            # fixme
            # warm_start_epochs 取多少
            only_fc_unfreezed_flag = False
            if ep < warm_start_epochs:
                if not only_fc_unfreezed_flag:
                    # freeze all layers except fc layer
                    self.freeze_model()
                    self.unfreeze_last_k_blocks(1)
                    only_fc_unfreezed_flag = True
                optimizer = self.fc_optimizer
            else:
                if only_fc_unfreezed_flag:
                    # unfreeze all layers
                    self.unfreeze_model()
                optimizer = self.all_optimizer

            # show current learning rate
            logger.debug(f"lr: {self.current_lr}")

            training_acc, running_loss = 0, .0
            start_time = time.perf_counter()

            for index, data in enumerate(self._train_loader):
                batch_running_loss, batch_training_acc = self.step_batch(data[0], data[1], optimizer)

                training_acc += batch_training_acc
                running_loss += batch_running_loss

                # warm up learning rate
                if ep <= self._warm_up_epochs:
                    self.warm_up_scheduler.step()

                if index % batch_number == batch_number - 1:
                    end_time = time.perf_counter()

                    acc = self.test()
                    average_train_loss = (running_loss / batch_number)
                    average_train_accuracy = training_acc / batch_number
                    epoch_cost_time = end_time - start_time

                    # write loss, time, test_acc, train_acc to tensorboard
                    if hasattr(self, "summary_writer"):
                        self.summary_writer: SummaryWriter
                        self.summary_writer.add_scalar("train loss", average_train_loss, ep)
                        self.summary_writer.add_scalar("train accuracy", average_train_accuracy, ep)
                        self.summary_writer.add_scalar("test accuracy", acc, ep)
                        self.summary_writer.add_scalar("time per epoch", epoch_cost_time, ep)

                    logger.info(
                        f"epoch: {ep}   loss: {average_train_loss:.6f}   train accuracy: {average_train_accuracy}   "
                        f"test accuracy: {acc}   time: {epoch_cost_time:.2f}s")

                    if best_acc < acc:
                        best_acc = acc
                        self._save_best_model(save_path, ep, acc)

            self._save_checkpoint(ep, best_acc)

        logger.info("finished training")
        logger.info(f"best accuracy on test set: {best_acc}")


class DatasetWithRobustFeatureRepresentations(Dataset):

    def __init__(self, origin_train_loader: DataLoader, model: SupportedModuleType, device: torch.device):
        """extend origin dataset with robust feature representations

        Args:
            origin_train_loader: origin train loader
            model: untrained robust model
        """
        logger.info("precalculate robust feature representations")

        self._batch_number = len(origin_train_loader.dataset)

        inputs_tensor_list = []
        feature_representations_tensor_list = []
        labels_tensor_list = []

        model.eval()
        for inputs, labels in origin_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs_tensor_list.append(inputs.detach().cpu())
            model(inputs)
            robust_feature_representations = model.get_feature_representations()
            # todo
            # cost lots of gpu
            feature_representations_tensor_list.append(robust_feature_representations.detach().cpu())
            labels_tensor_list.append(labels.detach().cpu())

        self._inputs = torch.cat(inputs_tensor_list, dim=0)
        self._feature_representations = torch.cat(feature_representations_tensor_list, dim=0)
        self._labels = torch.cat(labels_tensor_list, dim=0)

        logger.debug(f"dataset inputs shape: {self._inputs.shape}")
        logger.debug(f"dataset feature representations shape: {self._feature_representations.shape}")
        logger.debug(f"dataset labels shape: {self._labels.shape}")

        model.train()

        logger.info("calculate done")

    def __getitem__(self, idx) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        inputs = self._inputs[idx]
        feature_representations = self._feature_representations[idx]
        labels = self._labels[idx]

        return (inputs, feature_representations), labels

    def __len__(self):
        return self._batch_number

if __name__ == '__main__':
    from src.networks import wrn34_10
    from src.utils import get_cifar_test_dataloader, get_cifar_train_dataloader, logger
    from src import settings

    logger.change_log_file(settings.log_dir / "tmp.log")

    model = wrn34_10(num_classes=10)
    teacher_model_path = "/home/aiandiot/usb/qiufeng/TransformRobust/trained_models/cifar100_pgd7_train-best"
    teacher_state_dict = torch.load(teacher_model_path, map_location=settings.device)
    teacher_state_dict["fc.weight"] = torch.rand_like(model.fc.weight)
    teacher_state_dict["fc.bias"] = torch.rand_like(model.fc.bias)
    model.load_state_dict(teacher_state_dict)
    model.to(settings.device)

    # origin_loader = get_cifar_test_dataloader("cifar10")
    origin_loader = get_cifar_train_dataloader("cifar10")

    test_dataset = DatasetWithRobustFeatureRepresentations(origin_loader, model)
    test_loader = DataLoader(test_dataset, batch_size=origin_loader.batch_size, num_workers=origin_loader.num_workers,
                             shuffle=True)

    print(len(test_loader))
    print(len(origin_loader))

    model.eval()
    for (inputs, feature_representations), _ in test_loader:
        inputs = inputs.to(settings.device)
        feature_representations = feature_representations.to(settings.device)
        model(inputs)
        running_feature_representations = model.get_feature_representations()
        print(torch.mean(torch.norm(feature_representations - running_feature_representations, p=1, dim=1)))

