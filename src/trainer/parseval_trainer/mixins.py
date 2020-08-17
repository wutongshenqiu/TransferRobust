import torch
from torch import nn

from typing import Tuple, Union
import math

from src.utils import logger
from ..retrain_trainer import WRN34Block
from src.networks import ParsevalBasicBlock


class ParsevalTrainerMixin:
    model: nn.Module
    _device: Union[str, torch.device]
    _blocks: WRN34Block

    def sum_layers_constrain(self) -> torch.Tensor:
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
            _scaling = 1
            for a, b in zip(kernel_size, stride):
                _scaling *= math.ceil(a / b)

            return _scaling

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
                layer: Union[nn.Module, ParsevalBasicBlock]
                if isinstance(layer, ParsevalBasicBlock):
                    for residual_layer in layer:
                        if isinstance(residual_layer, nn.Conv2d):
                            self._layers_needed_constrain["conv"].append(residual_layer)
                if isinstance(layer, nn.Linear):
                    self._layers_needed_constrain["fc"].append(layer)

        # if k = 17, we should add the first convolutional layer
        if k == 17:
            self._layers_needed_constrain["conv"].append(self.model.conv1)

        logger.debug(f"layers needed constrain: \n{self._layers_needed_constrain}")
