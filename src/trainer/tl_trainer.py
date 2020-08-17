from typing import List

import torch
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader

from . import NormalTrainer
from src.utils import logger


class CIFARTLTrainer(NormalTrainer):
    def __init__(self, teacher_model_path: str,
                 model: Module, train_loader: DataLoader,
                 test_loader: DataLoader, checkpoint_path: str = None):
        teacher_state_dict = self._load_teacher_state_dict(teacher_model_path)
        self._reshape_teacher_fc_layer(model, teacher_state_dict)
        model.load_state_dict(teacher_state_dict)
        self._reinitialize_layers_weight(model, ["fc"])
        self._freeze_untrained_layers(model, ["fc"])
        super(NormalTrainer, self).__init__(model, train_loader, test_loader, checkpoint_path)

    def _load_teacher_state_dict(self, teacher_model_path: str):
        return torch.load(teacher_model_path)

    def _reshape_teacher_fc_layer(self, model, state_dict) -> None:
        state_dict["fc.weight"] = torch.rand_like(model.fc.weight)
        if state_dict.get("fc.bias") is not None:
            state_dict["fc.bias"] = torch.rand_like(model.fc.bias)

    def _reinitialize_layers_weight(self, model: Module, layer_list: List[str]) -> None:
        for layer in layer_list:
            _layer = getattr(model, layer)
            if hasattr(_layer, "reset_parameters"):
                _layer.reset_parameters()

    def _freeze_untrained_layers(self, model, trained_layers: List[str]) -> None:
        for p in model.parameters():
            p.requires_grad = False

        for layer in trained_layers:
            _layer = getattr(model, layer)
            for p in _layer.parameters():
                p.requires_grad = True

        # print trainable layers
        logger.debug("trainable layers")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.debug(f"name: {name}, size: {param.size()}")