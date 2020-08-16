from .wrn import wrn34_10, WideResNet
from .parseval_wrn import (parseval_retrain_wrn34_10, ParsevalBasicBlock,
                           parseval_normal_wrn34_10, ParsevalWideResNet)

from typing import Union
SupportedModuleType = Union[WideResNet, ParsevalWideResNet]
