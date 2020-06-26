from __future__ import print_function
from typing import Any, Callable, Union

import torch
from torch import Tensor
import torch.nn.functional as functional
import numpy as np


class FGSMAttack:

    def __init__(self, model: torch.nn.Module, clip_min=0, clip_max=1,
                 epsilon=0.3, p=-1, loss_function: Callable[[Any], Tensor] = None
                 ):
        self.model = model
        self.min = clip_min
        self.max = clip_max
        self.epsilon = epsilon
        self.p = p
        if loss_function is None:
            self.loss_function = functional.cross_entropy

    def calc_perturbation(self, x: Tensor, target: Tensor) -> Tensor:
        xt = x
        xt.requires_grad = True
        y_hat: Tensor = self.model(xt)
        # todo
        # this may interrupt other grad
        self.model.zero_grad()
        loss: Tensor = self.loss_function(y_hat, target)
        loss.backward()
        grad = xt.grad.data
        sign = grad.sign()
        return xt + self.epsilon * sign

    def print_params(self):
        print(f"epsilon: {self.epsilon}, loss: {self.loss_function.__name__}")


class PGDAttack:

    def __init__(self, model: torch.nn.Module, clip_min=0, clip_max=1,
                 random_init=True, epsilon=0.3, alpha=0.01, iter_num=10, p=-1,
                 loss_function: Callable[[Any], Tensor] = None
                 ):

        self.min = clip_min
        self.max = clip_max
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.p = p
        self.random_init = random_init
        self.iter_num = iter_num
        if loss_function is None:
            self.loss_function = functional.cross_entropy

    def random_delta(self, delta: Tensor) -> Tensor:
        if self.p == -1:
            delta.uniform_(-1, 1)
            delta = delta * self.epsilon
        else:
            pass
        return delta

    def calc_perturbation(self, x: Tensor, target: Tensor) -> Tensor:
        delta = torch.zeros_like(x)
        if self.random_init:
            delta = self.random_delta(delta)
        xt = x + delta
        xt.requires_grad = True

        for it in range(self.iter_num):
            y_hat = self.model(xt)  # type: Tensor
            loss = self.loss_function(y_hat, target)  # type: Tensor

            self.model.zero_grad()
            loss.backward()
            if self.p == -1:
                grad_sign = xt.grad.detach().sign()
                xt.data = xt.detach() + self.alpha * grad_sign
                xt.data = torch.clamp(xt - x, -self.epsilon, self.epsilon) + x
                xt.data = torch.clamp(xt.detach(), self.min, self.max)
            else:
                pass

            xt.grad.data.zero_()

        return xt

    def print_params(self):
        print(f"iter_num: {self.iter_num}, epsilon: {self.epsilon}, loss: {self.loss_function.__name__}")


class DeepFoolAttack:

    def __init__(self, model: torch.nn.Module, *,
                 num_classes=10, overshoot=0.02, max_iter=100,
                 loss_function: Callable[[Any], Tensor] = None):
        self.model = model
        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iter = max_iter
        if loss_function is None:
            self.loss_function = functional.cross_entropy

    # old vision
    # def calc_perturbation(self, x: Tensor, origin_label: Tensor):
    #
    #     def cal_grad(x: Tensor, target: Tensor):
    #         # 这里有点坑，先设置为叶子节点
    #         x = x.clone().detach()
    #         x.requires_grad_(True)
    #         self.model.zero_grad()
    #         output = self.model(x)
    #         loss = self.loss_function(output, target)
    #         loss.backward()
    #         return x.grad.data
    #
    #     f = self.model(x).data.cpu().numpy().flatten()
    #     # todo
    #     # not very understand
    #     labels = np.argsort(f)[::-1]
    #     labels = labels[0:self.num_classes]
    #
    #     grad = cal_grad(x, origin_label)
    #     r_tot = torch.zeros_like(x)
    #     k_i = origin_label
    #     loop_i = 0
    #
    #     # 这也太坑了吧
    #     # tensor.clone() 后不是叶子
    #     perturbation_image = x.clone().detach()
    #     perturbation_image.requires_grad_(True)
    #
    #     while k_i == origin_label and loop_i < self.max_iter:
    #         w = np.inf
    #         pert = np.inf
    #
    #         for k in labels:
    #             if k == origin_label:
    #                 continue
    #
    #             # 标签 k 对应的梯度
    #             k = torch.tensor([k], dtype=torch.long)
    #             grad_k = cal_grad(perturbation_image, k)
    #             w_k = grad_k - grad
    #             f_k = f[k] - f[origin_label]
    #
    #             w_k_norm = np.linalg.norm(w_k.flatten())
    #             pert_k = np.abs(f_k) / w_k_norm
    #
    #             if pert_k < pert:
    #                 pert = pert_k
    #                 w = w_k
    #
    #         r_i = w * (pert + 1e-4) / np.linalg.norm(w)
    #         r_tot += r_i
    #         perturbation_image = perturbation_image + (1 + self.overshoot) * r_tot
    #
    #         f = self.model(perturbation_image)
    #         k_i = torch.argmax(f).item()
    #         f = f.data.cpu().numpy().flatten()
    #         grad = cal_grad(perturbation_image, origin_label)
    #
    #         loop_i += 1
    #
    #     return perturbation_image

    # baidu vision
    def calc_perturbation(self, x: Tensor, origin_label: Tensor) -> Tensor:

        def cal_grad(x: Tensor, target: Tensor):
            # 这里有点坑，先设置为叶子节点
            x = x.clone().detach()
            x.requires_grad_(True)
            self.model.zero_grad()
            output = self.model(x)
            loss = self.loss_function(output, target)
            loss.backward()
            return x.grad.data

        f = self.model(x).data.cpu().numpy().flatten()
        # todo
        # not very understand
        labels = np.argsort(f)[::-1]
        labels = labels[0:self.num_classes]

        grad = cal_grad(x, origin_label)
        # r_tot = torch.zeros_like(x)
        k_i = origin_label
        loop_i = 0

        # 这也太坑了吧
        # tensor.clone() 后不是叶子
        perturbation_image = x.clone().detach()
        perturbation_image.requires_grad_(True)

        while k_i == origin_label and loop_i < self.max_iter:
            w = np.inf
            w_norm = np.inf
            pert = np.inf

            for k in labels:
                if k == origin_label:
                    continue

                # 标签 k 对应的梯度
                k = torch.tensor([k], dtype=torch.long)
                grad_k = cal_grad(perturbation_image, k)
                w_k = grad_k - grad
                f_k = f[k] - f[origin_label]

                w_k_norm = np.linalg.norm(w_k.flatten()) + 1e-8
                pert_k = (1e-8 + np.abs(f_k)) / w_k_norm

                if pert_k < pert:
                    pert = pert_k
                    w = w_k
                    w_norm = w_k_norm

            r_i = w * pert / w_norm
            perturbation_image = perturbation_image + (1 + self.overshoot) * r_i

            f = self.model(perturbation_image)
            k_i = torch.argmax(f).item()
            f = f.data.cpu().numpy().flatten()
            grad = cal_grad(perturbation_image, origin_label)

            loop_i += 1

        return perturbation_image


if __name__ == '__main__':

    # 为啥效果会这么差？
    from lenet import MNISTModel
    from utils import get_mnist_testing_data, test_model, grey_to_img

    model = MNISTModel()
    test_loader = get_mnist_testing_data(batch_size=1)
    # ignore dropout
    model.eval()
    model.load_state_dict(torch.load("./model/mnist"))

    # test_model(model, test_loader)

    test_img: Tensor
    test_label: Tensor
    n = 0
    for img, label in test_loader:
        if n == 8765:
            test_img = img
            test_label = label
            break
        n += 1
    grey_to_img(test_img[0][0])
    print(f"origin label: {test_label.item()}")
    print(f"model output: {torch.argmax(model(test_img)).item()}")

    # attacker = DeepFoolAttack(model)
    # adv_img = attacker.cal_perturbation(test_img, test_label)

    attacker = DeepFoolAttack(model)
    adv_img = attacker.calc_perturbation(test_img, test_label)

    grey_to_img(adv_img[0][0].detach().numpy())
    print(f"adversarial img: {torch.argmax(model(adv_img)).item()}")

