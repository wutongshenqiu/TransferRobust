from src.utils import get_mnist_test_dataloader, logger, get_mnist_test_dataloader_one_channel
from src.networks import resnet18
from src import settings
from .attack import LinfPGDAttack

import torch

import matplotlib.pyplot as plt

EPSILON = 0.1

mnist_attack_params = {
    "random_init": 1,
    "epsilon": EPSILON,
    "step_size": EPSILON/4,
    "num_steps": 100,
    "dataset_name": "mnist",
}


DEVICE = "cuda: 0"


def prepare_img(img: torch.Tensor):
    std = torch.as_tensor([0.19803032, 0.20101574, 0.19703609])[:, None, None]
    mean = torch.as_tensor([0.4376817, 0.4437706, 0.4728039])[:, None, None]
    print(std, mean)
    fig = (img * std + mean).clone().detach()
    # fig = fig * 255
    return fig.numpy().transpose(1, 2, 0)


if __name__ == "__main__":
    logger.change_log_file(settings.log_dir / "tmp.log")

    test_loader = get_mnist_test_dataloader(batch_size=1, shuffle=False)
    # test_loader = get_mnist_test_dataloader_one_channel(batch_size=10, shuffle=False)
    model = resnet18(num_classes=10).to(DEVICE)
    model.load_state_dict(
        torch.load("trained_models/normalization_svhn_tl_svhn_pgd7_train_blocks1-best", map_location=DEVICE))
    model.eval()

    attacker = LinfPGDAttack(model, **mnist_attack_params)
    for inputs, labels in test_loader:
        raw_img = prepare_img(inputs[0])
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        print(f"original labels: {labels}")
        _, pred_labels = model(inputs).max(1)
        print(f"predicted labels: {pred_labels}")
        adv_inputs = attacker.calc_perturbation(inputs, labels)
        _, adv_labels = model(adv_inputs).max(1)
        print(f"adversarial labels: {adv_labels}")
        adv_inputs = adv_inputs.cpu().detach()
        plt.subplot(1,2,1)
        plt.imshow(raw_img)
        plt.subplot(1,2,2)
        plt.imshow(prepare_img(adv_inputs[0]))
        plt.show()

        break


