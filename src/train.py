import torch

from .networks import wrn34_10

from .trainer import (ADVTrainer, RetrainTrainer,
                      CIFARTLTrainer, RobustPlusRegularizationTrainer,
                      ParsevalRetrainTrainer, parseval_wrn34_10)

from . import settings
from .utils import get_cifar_test_dataloader, get_cifar_train_dataloader, logger
from .attack import LinfPGDAttack, attack_params

# if the batch_size and model structure is fixed, this may accelerate the training process
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    import time
    logger.info(settings)
    model = parseval_wrn34_10(num_classes=10)
    # tranform learning
    # model = wrn34_10(num_classes=10)
    # trainer = CIFARTLTrainer(
    #     teacher_model_path="../trained_models/cifar100_wrn34_10-best",
    #     model=model,
    #     train_loader=get_cifar_train_dataloader("cifar10"),
    #     test_loader=get_cifar_test_dataloader("cifar10"),
    #     checkpoint_path="../checkpoint/checkpoint.pth"
    # )

    # fixme
    # still have bugs
    # trainer = ARTTrainer(
    #     model, get_cifar_train_dataloader("cifar10", normalize=False),
    #     get_cifar_test_dataloader("cifar10", normalize=False),
    #     attacker="ProjectedGradientDescent",
    #     params=attack_params.get("ProjectedGradientDescent"),
    #     dataset_mean=CIFAR10_TRAIN_MEAN,
    #     dataset_std=CIFAR10_TRAIN_STD,
    #     checkpoint_path="./checkpoint/checkpoint.pth"
    # )

    #
    # trainer = ADVTrainer(
    #     # todo
    #     # !!! 这里不能使用 normalize，因为 attack 的实现里面没有考虑 normalize
    #     # 那ART训练又是为什么呢？
    #     model, get_cifar_train_dataloader(),
    #     get_cifar_test_dataloader(),
    #     attacker=LinfPGDAttack,
    #     params=attack_params.get("LinfPGDAttack"),
    #     checkpoint_path="./checkpoint/checkpoint_wrn34.pth"
    # )

    # retrain
    # model.load_state_dict(torch.load("./trained_models/cifar10_robust_plus_regularization_k6_1-best", map_location=settings.device))
    # trainer = RetrainTrainer(
    #     k=6,
    #     model=model,
    #     train_loader=get_cifar_train_dataloader("cifar10"),
    #     test_loader=get_cifar_test_dataloader("cifar10"),
    #     checkpoint_path="./checkpoint/retrain_cifar10_robust_plus_regularization_k6_1.pth"
    # )

    teacher_model_path = "./trained_models/cifar10_robust_plus_regularization_k6_1-best"
    model.load_state_dict(torch.load(teacher_model_path, map_location=settings.device))
    logger.debug(f"teacher model: {teacher_model_path}")
    # parseval retrain
    trainer = ParsevalRetrainTrainer(
        beta=0.0003,
        k=6,
        model=model,
        train_loader=get_cifar_train_dataloader(),
        test_loader=get_cifar_test_dataloader(),
        checkpoint_path="./checkpoint/parseval_retrain_cifar10_robust_plus_regularization_k6_1.pth"
    )

    # robust plus regularization
    # _k = 6
    # _lambda = 1
    # trainer = RobustPlusRegularizationTrainer(
    #     k=_k,
    #     _lambda=_lambda,
    #     model=model,
    #     train_loader=get_cifar_train_dataloader("cifar10"),
    #     test_loader=get_cifar_test_dataloader("cifar10"),
    #     attacker=LinfPGDAttack,
    #     params=attack_params.get("LinfPGDAttack"),
    #     checkpoint_path=f"./checkpoint/cifar10_robust_plus_regularization_k{_k}_{_lambda}",
    #     # use sub directory to support multi SummaryWriter
    #     log_dir=f"./runs/lambda_{_lambda}",
    # )

    trainer.train("./trained_models/parseval_retrain_cifar10_robust_plus_regularization_k6_1")
