import torch
from torch.utils.data import DataLoader

from .mixins import ReshapeTeacherFCLayerMixin
from ..normal_trainer import NormalTrainer
from ..retrain_trainer import ResetBlockMixin, FreezeModelMixin, WRN34Block
from src.networks import SupportedModuleType
from src.utils import logger


class TransformLearningTrainer(NormalTrainer, ResetBlockMixin, FreezeModelMixin, ReshapeTeacherFCLayerMixin):

    def __init__(self, k: int, teacher_model_path: str,
                 model: SupportedModuleType, train_loader: DataLoader,
                 test_loader: DataLoader, checkpoint_path: str = None):
        super().__init__(model, train_loader, test_loader, checkpoint_path)

        teacher_state_dict = torch.load(teacher_model_path, map_location=self._device)
        self.reshape_teacher_fc_layer(teacher_state_dict)
        logger.info(f"load from teacher model: \n {teacher_model_path}")
        self.model.load_state_dict(teacher_state_dict)

        self._blocks = WRN34Block(model)

        self.freeze_model()
        self.reset_and_unfreeze_last_k_blocks(k)


        



if __name__ == '__main__':
    from src.networks import wrn34_10
    from src.utils import get_cifar_test_dataloader, get_cifar_train_dataloader

    trainer = TransformLearningTrainer(
        k=6,
        teacher_model_path="./trained_models/cifar100_pgd7_train-best",
        model=wrn34_10(num_classes=10),
        train_loader=get_cifar_train_dataloader("cifar10"),
        test_loader=get_cifar_test_dataloader("cifar10"),
    )