import torch

from .networks import wrn34_10, parseval_retrain_wrn34_10, resnet18, parseval_resnet18

from .trainer import (ADVTrainer, RetrainTrainer, TransferLearningTrainer,
                      ParsevalTransferLearningTrainer, RobustPlusSingularRegularizationTrainer,
                      ParsevalRetrainTrainer, ParsevalNormalTrainer,
                      RobustPlusAllRegularizationTrainer, NormalTrainer,
                      LWFTransferLearningTrainer)

from . import settings
from .utils import (get_cifar_test_dataloader, get_cifar_train_dataloader, logger,
                    get_subset_cifar_train_dataloader, get_svhn_train_dataloder,
                    get_mnist_test_dataloader, get_mnist_train_dataloader,
                    get_svhn_test_dataloader)

from .attack import LinfPGDAttack, attack_params

# if the batch_size and model structure is fixed, this may accelerate the training process
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    from src.cli.utils import get_model, get_train_dataset, get_test_dataset
    model = "wrn34"
    dataset = "cifar100"
    lambda_ = 0.005
    random_init = True
    epsilon = 8 / 255
    num_steps = 7
    step_size = 2 / 255
    num_classes = 100
    save_name = f"cartl_{model}_{dataset}_all_{lambda_}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    params = {
        "random_init": random_init,
        "epsilon": epsilon,
        "step_size": step_size,
        "num_steps": num_steps,
        "dataset_name": dataset
    }
    trainer = RobustPlusAllRegularizationTrainer(
        _lambda=lambda_,
        model=get_model(model, num_classes),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        attacker=LinfPGDAttack,
        params=params,
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")