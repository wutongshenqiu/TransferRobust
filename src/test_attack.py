import time
import torch
from typing import Callable
from utils import get_test_dataloader
import argparse
from models import resnet
from attack import FGSMAttack, PGDAttack
import os
import json

epsilons = [.01, .02, .03, .04, .05, .06, .07, .08, .09, .1, .15, .2]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 200


def test_attack(model: torch.nn.Module, test_loader, attacker: Callable):
    incorrectness_list = []
    for eps in epsilons:
        start_time = time.process_time()
        _attacker = attacker(model, epsilon=eps)

        correct = 0
        adv_examples = []
        la_correct = 0
        incorrect = 0
        counter = 0
        for data, label in test_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)

            if counter % 10 == 0:
                print(
                    f"{type(_attacker).__name__}@{_attacker.epsilon}: Fig. #{counter * BATCH_SIZE}  Left: {len(test_loader) * BATCH_SIZE - counter * BATCH_SIZE}")
            counter += 1
            # (1, 200) tensor
            init_pred = model(data).max(1)[1]
            origin_match = (init_pred == label)
            for j in range(len(origin_match)):
                if not origin_match[j].item():
                    continue
                # the number of origin_cifar100 correct
                la_correct += 1

            data_distorred = _attacker.calc_perturbation(data, target=label)

            adv_pred = model(data_distorred).max(1)[1]
            adv_match = (adv_pred == label)

            for j in range(len(adv_match)):
                if not origin_match[j].item():
                    continue
                if adv_match[j].item():
                    correct += 1
                    if _attacker.epsilon == 0 and len(adv_examples) < 5:
                        adv_ex = data_distorred[j].squeeze().detach().cpu().numpy()
                        adv_examples.append((label[j].item(), adv_pred[j].item(), adv_ex))
                else:
                    incorrect += 1
                    if len(adv_examples) < 5:
                        adv_ex = data_distorred[j].squeeze().detach().cpu().numpy()
                        adv_examples.append((label[j].item(), adv_pred[j].item(), adv_ex))

        acc = correct / float(la_correct)
        incorrectness_list.append(incorrect / la_correct)
        print(f"Epsilon: {_attacker.epsilon}\tTest Accuracy = {correct} / {float(la_correct)} = {acc}")
        print(f"La Correctness: {la_correct}")
        print(f"Incorrectness: {incorrect}")

        end_time = time.process_time()
        print(f"using time: {end_time - start_time:.4f} seconds")

    return incorrectness_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="a cli that supports train/tl/attack")
    parser.add_argument("--dataset", required=True, help="the dataset, supporting cifar10 and cifar100")
    parser.add_argument("--model_path", required=True, help="the path of the model")
    parser.add_argument("--save_path", default="", help="the path to save result")

    args = parser.parse_args()

    if args.dataset == "cifar10":
        model = resnet.resnet50(num_classes=10)
        test_loader = get_test_dataloader("cifar10", batch_size=BATCH_SIZE)
    elif args.dataset == "cifar100":
        model = resnet.resnet50()
        test_loader = get_test_dataloader("cifar100", batch_size=BATCH_SIZE)
    else:
        raise NotImplemented("only support cifar10 or cifar100")

    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE)

    for attacker in (FGSMAttack, PGDAttack):
        results = test_attack(model, test_loader, attacker)
        if len(args.save_path) != 0:
            with open(os.path.join(args.save_path, f"{attacker.__name__}.json"), "w", encoding="utf8") as f:
                json.dump({
                    "eps": epsilons,
                    "acc": results
                }, f)
