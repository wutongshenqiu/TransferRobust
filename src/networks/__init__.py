from typing import Union

from .wrn import wrn34_10, wrn28_10, WideResNet
from .lenet import LeNet
from .parseval_wrn import (parseval_retrain_wrn34_10,
                           parseval_normal_wrn34_10, ParsevalWideResNet)

from .resnet import ResNet, resnet18
from .parseval_resnet import ParsevalResNet, parseval_resnet18

# todo
# these will be used in utils
SupportedWideResnetType = Union[WideResNet, ParsevalWideResNet]
SupportedResnetType = Union[ResNet, ParsevalResNet]
SupportedAllModuleType = Union[SupportedWideResnetType, SupportedResnetType]

from .utils import Resnet18Block, make_blocks, WRNBlocks

