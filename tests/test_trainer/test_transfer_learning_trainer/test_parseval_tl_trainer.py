from src.trainer import ParsevalTransferLearningTrainer
from src.networks import (wrn34_10, parseval_retrain_wrn34_10,
                          resnet18, parseval_resnet18)
from src import settings
from src.utils import (get_cifar_test_dataloader, get_cifar_train_dataloader,
                       get_svhn_test_dataloader, get_svhn_train_dataloder)


BETA = 0.0003
WRN_TEACHER_MODEL_PATH = str(settings.model_dir / "cifar100_pgd7_train-best")
RESNET_TEACHER_MODEL_PATH = str(settings.model_dir / "svhn_pgd7_train-best")

def test_wrn_parseval_tl_trainer():
    test_loader = get_cifar_train_dataloader("cifar10")
    train_loader = get_cifar_test_dataloader("cifar10")
    model = parseval_retrain_wrn34_10(k=8, num_classes=10)
    trainer = ParsevalTransferLearningTrainer(
        beta=BETA,
        k=8,
        teacher_model_path=WRN_TEACHER_MODEL_PATH,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        checkpoint_path=str(settings.model_dir / "test_model.pth")
    )
    assert len(trainer._layers_needed_constrain["conv"]) == 12
    assert len(trainer._layers_needed_constrain["fc"]) == 1


def test_resnet_parseval_tl_trainer():
    test_loader = get_svhn_train_dataloder()
    train_loader = get_svhn_test_dataloader()
    model = parseval_resnet18(k=3, num_classes=10)
    trainer = ParsevalTransferLearningTrainer(
        beta=BETA,
        k=8,
        teacher_model_path=WRN_TEACHER_MODEL_PATH,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        checkpoint_path=str(settings.model_dir / "test_model.pth")
    )
    assert len(trainer._layers_needed_constrain["conv"]) == 4
    assert len(trainer._layers_needed_constrain["fc"]) == 1
