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
    import time
    # logger.info(settings)
    # time.sleep(3000)

    # tranform learning
    # model = resnet18(num_classes=10)
    # for k in range(1, 10):
    #     teacher_model_path = f"./trained_models/normal_svhn_resnet18-best"
    #     save_path = f"normalization_svhn_tl_normal_svhn_resnet18_blocks{k}"
    #     logger.change_log_file(settings.log_dir / f"{save_path}.log")
    #     trainer = TransferLearningTrainer(
    #         k=k,
    #         teacher_model_path=teacher_model_path,
    #         model=model,
    #         train_loader=get_mnist_train_dataloader(),
    #         test_loader=get_mnist_test_dataloader(),
    #         checkpoint_path=f"./checkpoint/{save_path}.pth"
    #     )
    #     trainer.train(f"./trained_models/{save_path}")


    # parseval tranform learning
    # map_beta = {1e-3: "1e-3", 6e-4: "6e-4", 3e-4: "3e-4"}
    # for _lambda in [0.01, 0.05, 0.1, 2]:
    #     for beta in [2e-3]:
    #         for k in [8]:
    #             teacher_model_path = f"cifar100_robust_plus_regularization_blocks{k}_lambda{_lambda}"
    #             save_path = f"normalization_cifar100_parseval_tl_{teacher_model_path}_beta{map_beta[beta]}"
    #             # save_path = f"parseval_tl_cifar100_robust_plus_regularization_blocks1-17(retrain{k})_lambda{_lambda}_beta{map_beta[beta]}"
    #             logger.change_log_file(f"{settings.log_dir}/{save_path}.log")
    #             model = parseval_retrain_wrn34_10(k=k, num_classes=10)
    #             trainer = ParsevalTransferLearningTrainer(
    #                 beta=beta,
    #                 k=k,
    #                 # teacher_model_path=f"./trained_models/cifar100_robust_plus_regularization_blocks{k}_lambda1-best",
    #                 teacher_model_path=f"./trained_models/{teacher_model_path}-best",
    #                 model=model,
    #                 train_loader=get_cifar_train_dataloader("cifar10"),
    #                 test_loader=get_cifar_test_dataloader("cifar10"),
    #                 checkpoint_path=f"./checkpoint/{save_path}.pth"
    #             )
    #             trainer.train(f"./trained_models/{save_path}")

    # pgd7 train
    # model = resnet18(num_classes=10)
    # params = attack_params.get("svhn")
    # logger.change_log_file(settings.log_dir / "svhn_pgd7_train.log")
    # trainer = ADVTrainer(
    #     # !!! 这里不能使用 normalize，因为 attack 的实现里面没有考虑 normalize
    #     # 那ART训练又是为什么呢？
    #     model=model,
    #     train_loader=get_svhn_train_dataloder(),
    #     test_loader=get_svhn_test_dataloader(),
    #     attacker=LinfPGDAttack,
    #     params=attack_params.get("LinfPGDAttack"),
    #     checkpoint_path="./checkpoint/svhn_pgd7_train.pth"
    # )
    # trainer.train("./trained_models/svhn_pgd7_train")

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

    # normal train
    # model = resnet18(num_classes=10)
    # save_path = f"normal_svhn_resnet18"
    # logger.change_log_file(settings.log_dir / save_path)
    # trainer = NormalTrainer(
    #     model=model,
    #     train_loader=get_svhn_train_dataloder(),
    #     test_loader=get_svhn_test_dataloader(),
    #     checkpoint_path=f"./checkpoint/{save_path}.pth"
    # )
    # trainer.train(f"./trained_models/{save_path}")


    # robust plus regularization
    params = {
        "random_init": 1,
        "epsilon": 8/255,
        "step_size": 2/255,
        "num_steps": 7,
    }

    for _lambda in [1]:
        for k in [8]:
            model = resnet18(num_classes=10)
            save_path = f"svhn_robust_plus_regularization_blocks{k}_lambda{_lambda}"
            log_file = f"{settings.log_dir}/{save_path}.log"
            logger.change_log_file(log_file)
            trainer = RobustPlusSingularRegularizationTrainer(
                k=k,
                _lambda=_lambda,
                model=model,
                train_loader=get_svhn_train_dataloder(),
                test_loader=get_svhn_test_dataloader(),
                attacker=LinfPGDAttack,
                params=params,
                checkpoint_path=f"./checkpoint/{save_path}.pth",
            )
            trainer.train(f"trained_models/{save_path}")

    # parseval normal train
    # from src.networks import parseval_normal_wrn34_10
    # trainer = ParsevalNormalTrainer(
    #     beta=0.0003,
    #     model=parseval_normal_wrn34_10(num_classes=10),
    #     train_loader=get_cifar_train_dataloader("cifar10"),
    #     test_loader=get_cifar_test_dataloader("cifar10"),
    #     checkpoint_path="./checkpoint/cifar10_parseval_normal_train.pth"
    # )

    # trainer.train(f"./trained_models/{save_path}")
    # trainer.train(f"./trained_models/parseval_tl_cifar100_pgd7_blocks{k}_lambda1_beta6e-4")
    # trainer.train(f"./trained_models/tl_cifar100_robust_plus_regularization_blocks{k}_lambda1")
    # trainer.train(f"./trained_models/cifar100_pgd7_train")

    # lwf transfer training
    # _lambda = 0.1
    #
    # for partition_ratio in [0.5, 0.2, 0.1]:
    #     save_path = f"normalization_cifar100_lwf_tl_pgd7_lambda{_lambda}_ratio{partition_ratio}"
    #     logger.change_log_file(settings.log_dir / f"{save_path}.log")
    #
    #     model = wrn34_10(num_classes=10)
    #     teacher_model_path = "./trained_models/cifar100_pgd7_train-best"
    #     trainer = LWFTransferLearningTrainer(
    #         _lambda=_lambda,
    #         teacher_model_path=teacher_model_path,
    #         model=model,
    #         train_loader=get_subset_cifar_train_dataloader(partition_ratio=partition_ratio, dataset="cifar10"),
    #         test_loader=get_cifar_test_dataloader("cifar10"),
    #         checkpoint_path=str(settings.root_dir / "checkpoint" / f"{save_path}.pth")
    #     )
    #
    #     trainer.train(f"trained_models/{save_path}")
