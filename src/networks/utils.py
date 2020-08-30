import torch
from src.networks import SupportedWideResnetType, SupportedResnetType


class WRN34Block:
    """divide wrn34 model into 17 blocks,
    details can be found in paper `ADVERSARIALLY ROBUST TRANSFER LEARNING`"""

    def __init__(self, model: SupportedWideResnetType):
        self.model = model
        self._set_block()

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

    def _set_block(self):
        for i in range(1, 18):
            setattr(self, f"block{i}", self.get_block(i))


class Resnet18Block:
    """divided resnet into 10 blocks

    Blocks:
        block1: conv1
        block2-3: residual block of conv2_x
        block4-5: residual block of conv3_x
        block6-7: residual block of conv4_x
        block8-9: residual block of conv5_x
        block10: fully connect layer
    """

    def __init__(self, model: SupportedResnetType):
        self.model = model
        self._set_blocks()

    def get_block(self, num: int):
        if num == 1:
            return self.model.conv1
        elif 2 <= num <= 9:
            conv_num, residual_num = divmod(num+2, 2)
            conv_block = getattr(self.model, f"conv{conv_num}_x")
            return conv_block[residual_num]
        elif num == 10:
            return torch.nn.Sequential(self.model.fc)

    def _set_blocks(self):
        for i in range(1, 11):
            setattr(self, f"block{i}", self.get_block(i))


if __name__ == '__main__':
    from .resnet import resnet18

    model = resnet18(num_classes=10)
    blocks = Resnet18Block(model)
    for block in range(1, 11):
        print(getattr(blocks, f"block{block}"))