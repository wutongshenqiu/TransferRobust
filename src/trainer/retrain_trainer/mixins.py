from typing import Union

from torch.nn import BatchNorm2d

from src.utils import logger
from src.networks import WRN34Block, Resnet18Block
from src.networks import SupportedAllModuleType


class ResetBlockMixin:
    model: SupportedAllModuleType
    _blocks: Union[WRN34Block, Resnet18Block]

    def reset_and_unfreeze_last_k_blocks(self, k: int):
        """reset and unfreeze layers in last k blocks

        Args:
            k: the last k blocks which will be retrained
        """
        total_blocks = self._blocks.get_total_blocks()
        logger.debug(f"model: {type(self.model).__name__}, blocks: {total_blocks}")
        for i in range(total_blocks, total_blocks-k, -1):
            block = getattr(self._blocks, f"block{i}")
            for layer in block:
                if hasattr(layer, "reset_parameters"):
                    logger.debug(f"reinitialize layer: {type(layer).__name__}")
                    layer.reset_parameters()
            for p in block.parameters():
                p.requires_grad = True

        logger.debug(f"unfreeze and reset last {k} blocks")
        logger.debug("trainable layers")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.debug(f"name: {name}, size: {param.size()}")

    def reset_last_k_blocks(self, k: int):
        """reset layers in last k blocks

        Args:
            k: the last k blocks which will be retrained
        """
        total_blocks = self._blocks.get_total_blocks()
        logger.debug(f"model: {type(self.model).__name__}, blocks: {total_blocks}")
        for i in range(total_blocks, total_blocks-k, -1):
            block = getattr(self._blocks, f"block{i}")
            for layer in block:
                if hasattr(layer, "reset_parameters"):
                    logger.debug(f"reinitialize layer: {type(layer).__name__}")
                    layer.reset_parameters()

        logger.debug(f"reset last {k} blocks")

    def unfreeze_last_k_blocks(self, k: int):
        """unfreeze layers in last k blocks

        Args:
            k: the last k blocks which will be retrained
        """
        total_blocks = self._blocks.get_total_blocks()
        logger.debug(f"model: {type(self.model).__name__}, blocks: {total_blocks}")
        for i in range(total_blocks, total_blocks-k, -1):
            block = getattr(self._blocks, f"block{i}")
            for p in block.parameters():
                p.requires_grad = True


class FreezeModelMixin:
    """freeze all parameters of model"""
    model: SupportedAllModuleType

    def freeze_model(self):
        for p in self.model.parameters():
            p.requires_grad = False

        logger.debug(f"all parameters of model are freezed")

    def unfreeze_model(self):
        for p in self.model.parameters():
            p.requires_grad = True

        logger.debug(f"all parameters of model are unfreezed")

    def freeze_bn_layer(self):
        # difference between modules and parameters
        # https://blog.paperspace.com/pytorch-101-advanced/
        for layer in self.model.modules():
            if isinstance(layer, BatchNorm2d):
                for p in layer.parameters():
                    p.requires_grad = False

        logger.debug("all batch norm layers are freezed")
