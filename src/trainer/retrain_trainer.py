from typing import List

import torch
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader

from trainer import NormalTrainer


class WRN34Block:
    def __init__(self, model: torch.nn.modules):
        self.model = model
        self._set_block()

    def get_block_number(self, num: int):
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
            setattr(self, f"block{i}", self.get_block_number(i))


class RetrainTrainer(NormalTrainer):
    def __init__(self, k: int, model: Module, train_loader: DataLoader,
                 test_loader: DataLoader, checkpoint_path: str = None):
        super(RetrainTrainer, self).__init__(model, train_loader, test_loader, checkpoint_path)
        self._blocks = WRN34Block(model)
        self.freeze_last_k_blocks(k)

    def freeze_last_k_blocks(self, k):
        for p in self.model.parameters():
            p.requires_grad = False

        for i in range(17, 17-k, -1):
            block = getattr(self._blocks, f"block{i}")
            for layer in block:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
            for p in block.parameters():
                p.requires_grad = True

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"name: {name}, size: {param.size()}")


if __name__ == '__main__':
    from networks.wrn import wrn34_10

    model = wrn34_10()

    for layer in model.block1.layer:
        print(layer)
    # print(model)
