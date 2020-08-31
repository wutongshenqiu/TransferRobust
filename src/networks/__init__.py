from .wrn import wrn34_10, WideResNet
from .parseval_wrn import (parseval_retrain_wrn34_10, ParsevalBasicBlock,
                           parseval_normal_wrn34_10, ParsevalWideResNet)

from .resnet import ResNet, resnet18
from .parseval_resnet import ParsevalResNet, parseval_resnet18

from .utils import WRN34Block

from typing import Union

SupportedWideResnetType = Union[WideResNet, ParsevalWideResNet]
SupportedResnetType = Union[ResNet, ParsevalResNet]
SupportedAllModuleType = Union[SupportedWideResnetType, SupportedResnetType]
