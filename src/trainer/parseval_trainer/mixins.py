import torch
from torch import nn

from typing import Tuple, Union
import math

from src.utils import logger
from src.networks import WRN34Block
from src.networks import ParsevalBasicBlock, SupportedWideResnetType


class ParsevalConstrainMixin:
    """provide `gather_constrain_layers` and `sum_layers_constrain` method"""
    model: SupportedWideResnetType
    _device: Union[str, torch.device]
    _blocks: WRN34Block

    def sum_layers_constrain(self) -> torch.Tensor:
        """sum constrain of layers that have been gathered in `gather_constrain_layers`"""
        return sum(
            map(self._fully_connect_constrain, self._layers_needed_constrain["fc"])
        ) + sum(
            map(self._convolutional_constrain, self._layers_needed_constrain["conv"])
        )

    def _fully_connect_constrain(self, layer: nn.Linear) -> torch.Tensor:
        # choose the lower dimension for output matrix
        if layer.weight.shape[0] <= layer.weight.shape[1]:
            # weight matrix * transpose of weight matrix
            wwt = torch.matmul(layer.weight, layer.weight.T)
        else:
            # transpose of weight matrix * weight matrix
            wwt = torch.matmul(layer.weight.T, layer.weight)

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
        # put it to gpu to accelerate matrix multiplication
        flatten_matrix = layer.weight.view(layer.out_channels, -1).to(self._device)
        # choose the lower dimension for output matrix
        if flatten_matrix.shape[0] <= flatten_matrix.shape[1]:
            # weight matrix * transpose of weight matrix
            wwt = torch.matmul(flatten_matrix, flatten_matrix.T)
        else:
            # transpose of weight matrix * weight matrix
            wwt = torch.matmul(flatten_matrix.T, flatten_matrix)

        scaling = calculate_scaling(layer.kernel_size, layer.stride)
        scaling_identity_matrix = torch.eye(wwt.shape[0], device=self._device) / scaling
        # scaling_identity_matrix = scaling_identity_matrix.to(self._device)

        constrain_term = torch.norm(wwt - scaling_identity_matrix) ** 2
        # logger.debug(f"convolutional constrain: {constrain_term}")

        return constrain_term

    def gather_constrain_layers(self, k, ignore_first_conv: bool):
        """gather conv/fc layers in trainable layers to facilitate calculating regularization

        Args:
            k: the last k blocks which will be retrained
            ignore_first_conv: whether add first convolutional layer's constrain
                - in retrain, we should ignore first conv layer due to ignorance of first conv layer in `ResetBlockMixin`
                - in normal parseval train, we should not ignore first conv layer
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

        # todo
        if k == 17 and not ignore_first_conv:
            self._layers_needed_constrain["conv"].append(self.model.conv1)

        logger.debug(f"layers needed constrain: \n{self._layers_needed_constrain}")
        logger.debug(f"including {len(self._layers_needed_constrain['conv'])} convolutional layers, "
                     f"{len(self._layers_needed_constrain['fc'])} fully connect layers")

