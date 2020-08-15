"""parseval training

constrains:
    1. residual layer: f(x) + x -> 0.5x + 0.5f(x)
    2. fully connect layer: ||W * W^T - I||^2
    3. convolution layer: ||W * W^T - I / scaling||^2
"""

from typing import Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader

from . import RetrainTrainer

from src.utils import logger
from src import settings


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

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class ParsevalBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(ParsevalBasicBlock, self).__init__()
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


class ParsevalNetworkBlock(nn.Module):
    # record current residual block
    current_block: int = 0

    def __init__(self, k: int, nb_layers, in_planes, out_planes, stride, dropRate=0.0):
        """
        Args:
            k: the last k blocks which will be retrained
        """
        super(ParsevalNetworkBlock, self).__init__()
        self.layer = self._make_layer(k, in_planes, out_planes, nb_layers, stride, dropRate)

    @classmethod
    def _make_layer(cls, k: int, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            cls.current_block += 1
            # use ParsevalBasicBlock for residual block that needs retraining
            # 17 is the total block of wrn34
            if cls.current_block + k > 17:
                block = ParsevalBasicBlock
            else:
                block = BasicBlock
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class ParsevalWideResNet(nn.Module):
    def __init__(self, k: int, depth, num_classes, widen_factor=1, dropRate=0.0):
        """wide resnet for parseval training

        Args:
            k: the last k blocks which will be retrained
        """
        super(ParsevalWideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = ParsevalNetworkBlock(k, n, nChannels[0], nChannels[1], 1, dropRate)
        # 2nd block
        self.block2 = ParsevalNetworkBlock(k, n, nChannels[1], nChannels[2], 2, dropRate)
        # 3rd block
        self.block3 = ParsevalNetworkBlock(k, n, nChannels[2], nChannels[3], 2, dropRate)
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


def parseval_wrn34_10(k: int, num_classes=10):
    return ParsevalWideResNet(k, 34, num_classes, 10, 0)


class ParsevalRetrainTrainer(RetrainTrainer):
    def __init__(self, beta: float, k: int, model: Module, train_loader: DataLoader,
                 test_loader: DataLoader, checkpoint_path: str = None):
        """initialize retrain trainer

        Args:
            beta: retraction parameter
            k: the last k blocks which will be retrained
        """
        super(ParsevalRetrainTrainer, self).__init__(k, model, train_loader, test_loader, checkpoint_path)
        self._gather_constrain_layers(k)
        self._beta = beta

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        inputs, labels = inputs.to(self._device), labels.to(self._device)
        self.optimizer.zero_grad()

        outputs = self.model(inputs)

        constrain_term = self._sum_layers_constrain()
        logger.debug(f"batch constrain: {constrain_term}")
        loss = self.criterion(outputs, labels) + (self._beta / 2) * constrain_term
        loss.backward()
        self.optimizer.step()

        batch_training_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        batch_running_loss = loss.item()

        return batch_running_loss, batch_training_acc

    def _sum_layers_constrain(self) -> torch.Tensor:
        return sum(
            map(self._fully_connect_constrain, self._layers_needed_constrain["fc"])
        ) + sum(
            map(self._convolutional_constrain, self._layers_needed_constrain["conv"])
        )

    def _fully_connect_constrain(self, layer: nn.Linear) -> torch.Tensor:
        # weight matrix * transpose of weight matrix
        wwt = torch.matmul(layer.weight, layer.weight.T)
        identity_matrix = torch.eye(wwt.shape[0]).to(self._device)

        constrain_term = torch.norm(wwt - identity_matrix) ** 2
        # logger.debug(f"fully connect constrain: {constrain_term}")

        return constrain_term

    def _convolutional_constrain(self, layer: nn.Conv2d) -> torch.Tensor:
        def calculate_scaling(kernel_size: Tuple[int, ...], stride: Tuple[int, ...]) -> int:
            scaling = 1
            for a, b in zip(kernel_size, stride):
                scaling *= math.ceil(a / b)

            return scaling

        # flatten convolutional layer to c_out * (c_in * kernel_size) matrix
        flatten_matrix = layer.weight.view(layer.out_channels, -1)
        # weight matrix * transpose of weight matrix
        wwt = torch.matmul(flatten_matrix, flatten_matrix.T)

        scaling = calculate_scaling(layer.kernel_size, layer.stride)
        scaling_identity_matrix = torch.eye(wwt.shape[0]) / scaling
        scaling_identity_matrix = scaling_identity_matrix.to(self._device)

        constrain_term = torch.norm(wwt - scaling_identity_matrix) ** 2
        # logger.debug(f"convolutional constrain: {constrain_term}")

        return constrain_term

    def _gather_constrain_layers(self, k):
        """gather conv/fc layers in trainable layers to facilitate calculating regularization

        Args:
            k: the last k blocks which will be retrained
        """
        self._layers_needed_constrain = {
            "conv": [],
            "fc": []
        }
        logger.debug(f"model structure: \n{self.model}")
        for i in range(17, 17-k, -1):
            block = getattr(self._blocks, f"block{i}")
            for layer in block:
                layer: Union[Module, ParsevalBasicBlock]
                if isinstance(layer, ParsevalBasicBlock):
                    for residual_layer in layer:
                        if isinstance(residual_layer, nn.Conv2d):
                            self._layers_needed_constrain["conv"].append(residual_layer)
                if isinstance(layer, nn.Linear):
                    self._layers_needed_constrain["fc"].append(layer)

        logger.debug(f"layers needed regularization: \n{self._layers_needed_constrain}")


if __name__ == '__main__':
    from src.utils import get_cifar_train_dataloader
    from src.utils import get_cifar_test_dataloader

    parseval_retrain_trainer = ParsevalRetrainTrainer(
        beta=0.0003,
        k=6,
        model=parseval_wrn34_10(k=6, num_classes=10),
        train_loader=get_cifar_train_dataloader(),
        test_loader=get_cifar_test_dataloader(),
    )
