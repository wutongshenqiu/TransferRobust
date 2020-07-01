from models.resnet import resnet50, resnet34
from models.wrn import wrn34_10
from trainer import Trainer
from utils import get_cifar_training_dataloader, get_cifar_testing_dataloader, CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD, CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD
from art_usage import test_attack

from art.attacks import ProjectedGradientDescent, CarliniLInfMethod, CarliniL2Method, DeepFool
from art.classifiers import PyTorchClassifier
from art.utils import load_cifar10 

import torch

import numpy as np

import json

params = {
  "ProjectedGradientDescent": {
      # xz tql
      "eps": 8/255,
      "eps_step": 2/255,
      "batch_size": 128,
      "max_iter": 20,
      "num_random_init": 1
  },
  "CarliniLInfMethod": {
      "batch_size": 128,
      "binary_search_steps": 10,
      "max_iter": 100,
      "eps": 0.03
  },
  "CarliniL2Method": {
      "batch_size": 128,
      "binary_search_steps": 10,
      "max_iter": 10,
  },
  "DeepFool": {
      "max_iter": 75,
      "batch_size": 128,
      "epsilon": 0.02
  }
}


if __name__ == "__main__":
    model = wrn34_10(num_classes=100).to("cuda")
    model.load_state_dict(torch.load("./trained_models/cifar100_wrn34_10-best"))
    model.eval()

    test_num = 10000
    
    testing_loader = get_cifar_testing_dataloader("cifar100", batch_size=10000, normalize=False)
    data = next(iter(testing_loader))
    x_test = data[0][:test_num].numpy()
    y_test = data[1][:test_num].numpy()


    # (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
    # x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
    # x_test = x_test[:test_num]
    # y_test = y_test[:test_num]

    attacker = ProjectedGradientDescent

    
    mean = np.asarray(CIFAR100_TRAIN_MEAN).reshape((3, 1, 1))
    std = np.asarray(CIFAR100_TRAIN_STD).reshape((3, 1, 1))
    
    acc = test_attack(model, x_test, y_test, attacker, params.get(attacker.__name__), cuda_idx=0,
                # bug: ValueError: operands could not be broadcast together with shapes (640,3,32,32) (3,) 
                preprocessing=(mean, std), nb_classes=100)
