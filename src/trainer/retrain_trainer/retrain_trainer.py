from torch.utils.data import DataLoader

from src.trainer import NormalTrainer
from .mixins import ResetBlockMixin, FreezeModelMixin
from .utils import WRN34Block
from src.networks import SupportedModuleType


class RetrainTrainer(NormalTrainer, ResetBlockMixin, FreezeModelMixin):

    def __init__(self, k: int, model: SupportedModuleType, train_loader: DataLoader,
                 test_loader: DataLoader, checkpoint_path: str = None):
        """initialize retrain trainer

        Args:
            k: the last k blocks which will be retrained
        """
        super(RetrainTrainer, self).__init__(model, train_loader, test_loader, checkpoint_path)
        self._blocks = WRN34Block(model)
        self.freeze_model()
        self.reset_and_unfreeze_last_k_blocks(k)


if __name__ == '__main__':
    from src.networks import wrn34_10

    model = WRN34Block(wrn34_10())
    print(model.block15)
    # print(model)
