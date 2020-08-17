from torch import nn

from src.utils import logger
from .utils import WRN34Block
from src.networks import SupportedModuleType


class ResetBlockMixin:
    model: SupportedModuleType
    _blocks: WRN34Block

    def reset_and_unfreeze_last_k_blocks(self, k: int):
        """reset and unfreeze layers in last k blocks

        Args:
            k: the last k blocks which will be retrained
        """
        for i in range(17, 17-k, -1):
            block = getattr(self._blocks, f"block{i}")
            for layer in block:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
            for p in block.parameters():
                p.requires_grad = True

        logger.debug(f"unfreeze and reset last {k} blocks")
        logger.debug("trainable layers")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.debug(f"name: {name}, size: {param.size()}")


class FreezeModelMixin:
    """freeze all parameters of model"""
    model: SupportedModuleType

    def freeze_model(self):
        for p in self.model.parameters():
            p.requires_grad = False

        logger.debug(f"all parameters of model are freezed")
