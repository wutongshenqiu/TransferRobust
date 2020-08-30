"""get subset of pytorch dataset"""

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor

from typing import Tuple, Union, Dict, Any
import os
import math


def calculate_categories_size(data_loader: DataLoader) -> Dict[int, int]:
    """calculate data size of each category"""
    categories_size = dict()
    for _, labels in data_loader:
        for label in labels:
            label = label.item()
            try:
                categories_size[label] += 1
            except KeyError:
                categories_size[label] = 1

    return categories_size


class SubsetDataset(Dataset):

    def __init__(self, whole_data_loader: DataLoader, partition_ratio: float):
        """extend origin dataset with robust feature representations

        Args:
            whole_data_loader: original whole data loader,
            partition_ratio: proportion of partitioned subset

        Steps:
            1. calculate the original dataset size of each category
            2. calculate the dataset size after scaling according to the partition ratio
            3. generate subset dataset

        Notes:
            1. we presume that dataset size of each category of original data loader is same
            2. the original data loader should not be shuffled lest influence to the experiment
        """
        dataset_len = len(whole_data_loader.dataset) * partition_ratio
        whole_categories_size = list(calculate_categories_size(whole_data_loader).values())
        # check if all categories have the same dataset size
        if not all(x == whole_categories_size[0] for x in whole_categories_size):
            raise ValueError("size of categories are not same!")

        if not self._is_integer(dataset_len):
            raise ValueError("length of subset must be integer, choose the correct `partition ratio`!")
        else:
            self._dataset_len = math.ceil(dataset_len)

        subset_category_size = whole_categories_size[0] * partition_ratio
        if not self._is_integer(subset_category_size):
            raise ValueError("dataset size of subset category must be integer, choose the correct `partition ratio`!")
        else:
            subset_category_size = math.ceil(subset_category_size)

        inputs_tensor_list = []
        labels_tensor_list = []

        subset_categories_size = {}
        # todo
        # traverse the whole iterator may be slow
        for inputs, labels in whole_data_loader:
            for _input, label in zip(inputs, labels):
                label_item = label.item()
                try:
                    subset_categories_size[label_item] += 1
                except KeyError:
                    subset_categories_size[label_item] = 1
                if subset_categories_size[label_item] <= subset_category_size:
                    inputs_tensor_list.append(_input)
                    labels_tensor_list.append(label)

        self._inputs = torch.stack(inputs_tensor_list, dim=0)
        self._labels = torch.stack(labels_tensor_list, dim=0)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        inputs = self._inputs[idx]
        labels = self._labels[idx]

        return inputs, labels

    def __len__(self):
        return self._dataset_len

    @staticmethod
    def _is_integer(number: Any, threshold: float = 1e-10) -> bool:
        if abs(number-math.ceil(number)) > threshold:
            return False
        return True


if __name__ == '__main__':
    # train_dataset = datasets.CIFAR10(root=os.path.join(DATA_DIR, "CIFAR"),
    #                                  train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    #
    # train_loader = DataLoader(train_dataset, shuffle=False, num_workers=0, batch_size=32)
    # print(len(train_dataset))
    # print(len(train_loader))
    #
    # train_subset_dataset = SubsetDataset(train_loader, 0.5)
    #
    # train_dataset = datasets.CIFAR10(root=os.path.join(DATA_DIR, "CIFAR"),
    #                                  train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    #
    # # train_loader = DataLoader(train_dataset, shuffle=False, num_workers=0, batch_size=32)
    # #
    # # train_subset_dataset2 = SubsetDataset(train_loader, 0.2)
    # #
    # # for i in range(10000):
    # #     assert train_subset_dataset._labels[i] == train_subset_dataset2._labels[i]
    # train_subset_loader = DataLoader(train_subset_dataset, shuffle=True, num_workers=0, batch_size=32)
    # print(len(train_subset_dataset))
    #
    # print(calculate_categories_size(train_loader))
    # print(calculate_categories_size(train_subset_loader))
    pass