from typing import Optional

from torch.nn import Module
from torch.utils.data import DataLoader

from src.networks import (resnet18, wrn34_10,
                          parseval_retrain_wrn34_10, parseval_resnet18,
                          SupportedAllModuleType)

from src.utils import (get_cifar_test_dataloader, get_cifar_train_dataloader,
                       get_mnist_test_dataloader, get_mnist_train_dataloader,
                       get_svhn_test_dataloader, get_svhn_train_dataloder)


SupportNormalModelList = ['res18', 'wrn34']
SupportParsevalModelList = ['pres18', 'pwrn34']
SupportModelList = SupportNormalModelList + SupportParsevalModelList
DefaultModel = 'res18'

SupportDatasetList = ['cifar10', 'cifar100', 'mnist', 'svhn']
DefaultDataset = 'mnist'


def get_model(model: str, num_classes: int, k: Optional[int] = None) -> SupportedAllModuleType:
    if model not in SupportModelList:
        raise ValueError("model not supported")
    if model == 'res18':
        return resnet18(num_classes=num_classes)
    elif model == 'pres18':
        return parseval_resnet18(k=k, num_classes=num_classes)
    elif model == 'wrn34':
        return wrn34_10(num_classes=num_classes)
    elif model == 'pwrn34':
        return parseval_retrain_wrn34_10(k=k, num_classes=num_classes)


def get_train_dataset(dataset: str) -> DataLoader:
    if dataset not in SupportDatasetList:
        raise ValueError("dataset not supported")
    if dataset.startswith("cifar"):
        return get_cifar_train_dataloader(dataset=dataset)
    elif dataset == 'mnist':
        return get_mnist_train_dataloader()
    elif dataset.startswith('svhn'):
        # 'svhn': using mean and std of 'svhn'
        # 'svhn': using mean and std of 'cifar100'
        return get_svhn_train_dataloder(dataset_norm_type=dataset)


def get_test_dataset(dataset: str) -> DataLoader:
    if dataset not in SupportDatasetList:
        raise ValueError("dataset not supported")
    if dataset.startswith("cifar"):
        return get_cifar_test_dataloader(dataset=dataset)
    elif dataset == 'mnist':
        return get_mnist_test_dataloader()
    elif dataset == 'svhn':
        return get_svhn_test_dataloader()