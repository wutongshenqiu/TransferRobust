import torch

from .networks import wrn34_10, parseval_retrain_wrn34_10

from .trainer import (ADVTrainer, RetrainTrainer, TransformLearningTrainer,
                      ParsevalTransformLearningTrainer, RobustPlusRegularizationTrainer,
                      ParsevalRetrainTrainer, ParsevalNormalTrainer)

from . import settings
from .utils import get_cifar_test_dataloader, get_cifar_train_dataloader, logger
from .attack import LinfPGDAttack, attack_params

# if the batch_size and model structure is fixed, this may accelerate the training process
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    import time
    logger.info(settings)
    # time.sleep(3000)

    # tranform learning
    # model = wrn34_10(num_classes=10)
    # k = 8
    # trainer = TransformLearningTrainer(
    #     k=k,
    #     teacher_model_path=f"./trained_models/cifar100_robust_plus_regularization_blocks{k}_lambda1-best",
    #     model=model,
    #     train_loader=get_cifar_train_dataloader("cifar10"),
    #     test_loader=get_cifar_test_dataloader("cifar10"),
    #     # checkpoint_path="./checkpoint/tl_pgd7_block6.pth"
    #     checkpoint_path=f"./checkpoint/tl_cifar100_robust_plus_regularization_blocks{k}_lambda1.pth"
    # )


    # parseval tranform learning
    k = 6
    beta = 6e-4
    save_path = f"parseval_tl_cifar100_robust_plus_regularization_blocks{k}_lambda1_beta6e-4"
    model = parseval_retrain_wrn34_10(k=k, num_classes=10)
    trainer = ParsevalTransformLearningTrainer(
        beta=beta,
        k=k,
        teacher_model_path=f"./trained_models/cifar100_robust_plus_regularization_blocks{k}_lambda1-best",
        # teacher_model_path=f"./trained_models/cifar100_pgd7_train-best",
        model=model,
        train_loader=get_cifar_train_dataloader("cifar10"),
        test_loader=get_cifar_test_dataloader("cifar10"),
        checkpoint_path=f"./checkpoint/{save_path}.pth"
        # checkpoint_path=f"./checkpoint/parseval_tl_cifar100_pgd7_blocks{k}_lambda1.pth"
    )

    #
    # trainer = ADVTrainer(
    #     # !!! 这里不能使用 normalize，因为 attack 的实现里面没有考虑 normalize
    #     # 那ART训练又是为什么呢？
    #     model=wrn34_10(num_classes=100),
    #     train_loader=get_cifar_train_dataloader("cifar100"),
    #     test_loader=get_cifar_test_dataloader("cifar100"),
    #     attacker=LinfPGDAttack,
    #     params=attack_params.get("LinfPGDAttack"),
    #     checkpoint_path="./checkpoint/cifar100_pgd7_train.pth"
    # )

    # normal retrain
    # model.load_state_dict(torch.load("./trained_models/cifar10_robust_plus_regularization_k6_1-best", map_location=settings.device))
    # trainer = RetrainTrainer(
    #     k=6,
    #     model=model,
    #     train_loader=get_cifar_train_dataloader("cifar10"),
    #     test_loader=get_cifar_test_dataloader("cifar10"),
    #     checkpoint_path="./checkpoint/retrain_cifar10_robust_plus_regularization_k6_1.pth"
    # )

    # retrain blocks
    # k = 6
    # _lambda = 0.01
    # model = parseval_retrain_wrn34_10(k=k, num_classes=100)
    # teacher_model_path = f"./trained_models/cifar100_pgd7_train-best"
    # model.load_state_dict(torch.load(teacher_model_path, map_location=settings.device))
    # logger.debug(f"teacher model: {teacher_model_path}")
    # # parseval retrain
    # trainer = ParsevalRetrainTrainer(
    #     beta=0.0003,
    #     k=k,
    #     model=model,
    #     train_loader=get_cifar_train_dataloader(),
    #     test_loader=get_cifar_test_dataloader(),
    #     checkpoint_path=f"./checkpoint/parseval_retrain_cifar10_robust_plus_regularization_k{k}_{_lambda}.pth"
    # )

    # train with parseval network


    # robust plus regularization
    # k = 8
    # _lambda = 1
    # model = wrn34_10(num_classes=100)
    # trainer = RobustPlusRegularizationTrainer(
    #     k=k,
    #     _lambda=_lambda,
    #     model=model,
    #     train_loader=get_cifar_train_dataloader("cifar100"),
    #     test_loader=get_cifar_test_dataloader("cifar100"),
    #     attacker=LinfPGDAttack,
    #     params=attack_params.get("LinfPGDAttack"),
    #     checkpoint_path=f"./checkpoint/cifar100_robust_plus_regularization_blocks{k}_lambda{_lambda}.pth",
    #     # use sub directory to support multi SummaryWriter
    #     # log_dir=f"./runs/lambda_{_lambda}",
    # )

    # parseval normal train
    # from src.networks import parseval_normal_wrn34_10
    # trainer = ParsevalNormalTrainer(
    #     beta=0.0003,
    #     model=parseval_normal_wrn34_10(num_classes=10),
    #     train_loader=get_cifar_train_dataloader("cifar10"),
    #     test_loader=get_cifar_test_dataloader("cifar10"),
    #     checkpoint_path="./checkpoint/cifar10_parseval_normal_train.pth"
    # )

    trainer.train(f"./trained_models/{save_path}")
    # trainer.train(f"./trained_models/parseval_tl_cifar100_pgd7_blocks{k}_lambda1_beta6e-4")
    # trainer.train(f"./trained_models/tl_cifar100_robust_plus_regularization_blocks{k}_lambda1")
    # trainer.train(f"./trained_models/cifar100_pgd7_train")
