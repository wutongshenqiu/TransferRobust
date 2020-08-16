import torch
from src.networks import SupportedModuleType


class WRN34Block:
    """divide wrn34 model into 17 blocks,
    details can be found in paper `ADVERSARIALLY ROBUST TRANSFER LEARNING`"""

    def __init__(self, model: SupportedModuleType):
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
