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
from torch import Tensor

from typing import Tuple

from src.networks import SupportedModuleType


class LWFTransferLearningTrainer


class DatasetWithRobustFeatureRepresentations(Dataset):

    def __init__(self, origin_train_loader: DataLoader, model: SupportedModuleType):
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
            inputs, labels = inputs.to(settings.device), labels.to(settings.device)
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

