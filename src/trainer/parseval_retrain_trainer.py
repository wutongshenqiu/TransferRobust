"""questions
1. 是否需要加上固定的层
2. residual layer 这个应该是 \beta_1 f(x) + \beta_2 x
"""

from typing import Tuple, Union
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader

from . import RetrainTrainer

from src.utils import logger


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        # todo
        # convex combination
        former_out = (x if self.equalInOut else self.convShortcut(x)) * 0.5
        out = out * 0.5

        return torch.add(former_out, out)
        # return torch.add(x if self.equalInOut else self.convShortcut(x), out)

    def __iter__(self):
        return iter(
            [self.bn1, self.relu1, self.conv1, self.bn2, self.relu2, self.conv2]
        )


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def parseval_wrn34_10(num_classes=10):
    return WideResNet(34, num_classes, 10, 0)


class ParsevalRetrainTrainer(RetrainTrainer):
    def __init__(self, k: int, model: Module, train_loader: DataLoader,
                 test_loader: DataLoader, checkpoint_path: str = None):
        """initialize retrain trainer

        Args:
            k: the last k layers which will be retrained
        """
        super(ParsevalRetrainTrainer, self).__init__(k, model, train_loader, test_loader, checkpoint_path)
        self._gather_regularization_layers(k)

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        inputs, labels = inputs.to(self._device), labels.to(self._device)
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels) + self._regularization_constrain()
        loss.backward()
        self.optimizer.step()

        batch_training_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        batch_running_loss = loss.item()

        return batch_running_loss, batch_training_acc

    def _regularization_constrain(self) -> torch.Tensor:
        pass

    def _fully_connect_constrain(self, layer: nn.Linear) -> torch.Tensor:
        # weight matrix * transpose of weight matrix
        wwt = torch.matmul(layer.weight, layer.weight.T)
        identity_matrix = torch.eye(wwt.shape[0])
        return torch.norm(wwt - identity_matrix)

    def _convolutional_constrain(self, layer: nn.Conv2d) -> torch.Tensor:
        def calculate_scaling(kernel_size: Tuple[int, ...], stride: Tuple[int, ...]):
            scaling = 1
            for a, b in zip(kernel_size, stride):
                scaling *= math.ceil(a / b)

            return scaling

        scaling = calculate_scaling(layer.kernel_size, layer.stride)
        # todo
        # how to flatten convolutional layer to matrix

    def _gather_regularization_layers(self, k):
        """gather conv/fc layers in trainable layers to facilitate calculating regularization

        Args:
            k: the last k blocks which will be retrained
        """
        self._layers_needed_regularization = {
            "conv": [],
            "fc": []
        }
        for i in range(17, 17-k, -1):
            block = getattr(self._blocks, f"block{i}")
            for layer in block:
                layer: Union[Module, BasicBlock]
                if isinstance(layer, BasicBlock):
                    for residual_layer in layer:
                        if isinstance(residual_layer, nn.Conv2d):
                            self._layers_needed_regularization["conv"].append(residual_layer)
                if isinstance(layer, nn.Linear):
                    self._layers_needed_regularization["fc"].append(layer)

        logger.debug(f"layers needed regularization: \n{self._layers_needed_regularization}")


if __name__ == '__main__':
    # from src.utils import get_cifar_train_dataloader
    # from src.utils import get_cifar_test_dataloader
    #
    # parseval_retrain_trainer = ParsevalRetrainTrainer(
    #     k=6,
    #     model=parseval_wrn34_10(num_classes=10),
    #     train_loader=get_cifar_train_dataloader(),
    #     test_loader=get_cifar_test_dataloader(),
    # )
    #
    a = nn.Conv2d(3, 3, 3)
    print(a.stride)
    print(a.kernel_size)