from typing import Union

import torch

from .resnet import ResNet
from .parseval_resnet import ParsevalResNet
from .wrn import WideResNet
from .parseval_wrn import ParsevalWideResNet
from src.networks import SupportedWideResnetType, SupportedResnetType, SupportedAllModuleType


class WRN34Block:
    """divide wrn34 model into 17 blocks,
    details can be found in paper `ADVERSARIALLY ROBUST TRANSFER LEARNING`"""

    def __init__(self, model: SupportedWideResnetType):
        self.model = model
        self._set_block()

        self._total_blocks = 17

    def get_block(self, num: int):
        if 1 <= num <= 15:
            block_num, layer_num = divmod(num+4, 5)
            return torch.nn.Sequential(getattr(self.model, f"block{block_num}").layer[layer_num])
        elif num == 16:
            return torch.nn.Sequential(self.model.bn1, self.model.relu)
        elif num == 17:
            return torch.nn.Sequential(self.model.fc)
        else:
            raise ValueError(f"unexpected block number: {num}")

    def get_total_blocks(self) -> int:
        return self._total_blocks

    def _set_block(self):
        for i in range(1, 18):
            setattr(self, f"block{i}", self.get_block(i))


class Resnet18Block:
    """divided resnet into 9 blocks

    Blocks:
        block1-2: residual block of conv2_x
        block3-4: residual block of conv3_x
        block5-6: residual block of conv4_x
        block7-8: residual block of conv5_x
        block9: fully connect layer
    """

    def __init__(self, model: SupportedResnetType):
        self.model = model
        self._set_blocks()

        self._total_blocks = 9

    def get_block(self, num: int):
        if 1 <= num <= 8:
            conv_num, residual_num = divmod(num+3, 2)
            conv_block = getattr(self.model, f"conv{conv_num}_x")
            return conv_block[residual_num]
        elif num == 9:
            return torch.nn.Sequential(self.model.fc)
        else:
            raise ValueError(f"unexpected block number: {num}")

    def get_total_blocks(self) -> int:
        return self._total_blocks

    def _set_blocks(self):
        for i in range(1, 10):
            setattr(self, f"block{i}", self.get_block(i))


def make_blocks(model: SupportedAllModuleType) -> Union[WRN34Block, Resnet18Block]:
    if isinstance(model, WideResNet) or isinstance(model, ParsevalWideResNet):
        return WRN34Block(model)
    elif isinstance(model, ResNet) or isinstance(model, ParsevalResNet):
        return Resnet18Block(model)
    else:
        raise ValueError(f"model {type(model).__name__} is not supported to divide into blocks")


if __name__ == '__main__':
    from .resnet import resnet18
    from .parseval_resnet import parseval_resnet18
    from .wrn import wrn34_10
    from .parseval_wrn import parseval_retrain_wrn34_10

    model = resnet18(num_classes=10)
    blocks = make_blocks(model)
    print(blocks.block9)