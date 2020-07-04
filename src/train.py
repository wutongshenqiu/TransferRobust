import torch

from networks.wrn import wrn34_10

from trainer import ADVTrainer

# if the batch_size and model structure is fixed, this may accelerate the training process
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    from utils import get_cifar_test_dataloader, get_cifar_train_dataloader
    from art_utils import attack_params

    model = wrn34_10(num_classes=100)

    # tranform learning
    # model = wrn34_10(num_classes=10)
    # trainer = CIFARTLTrainer(
    #     teacher_model_path="./trained_models/cifar100_wrn34_10-best",
    #     model=model,
    #     train_loader=get_cifar_train_dataloader("cifar10"),
    #     test_loader=get_cifar_test_dataloader("cifar10"),
    #     checkpoint_path="./checkpoint.pth"
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
    #     checkpoint_path="./checkpoint.pth"
    # )

    from attack import LinfPGDAttack, attack_params

    trainer = ADVTrainer(
        # todo
        # !!! 这里不能使用 normalize，因为 attack 的实现里面没有考虑 normalize
        # 那ART训练又是为什么呢？
        model, get_cifar_train_dataloader(),
        get_cifar_test_dataloader(),
        attacker=LinfPGDAttack,
        params=attack_params.get("LinfPGDAttack"),
        checkpoint_path="./checkpoint/checkpoint_wrn34.pth"
    )

    # trainer.train("./trained_models/cifar100_robust_wrn34")