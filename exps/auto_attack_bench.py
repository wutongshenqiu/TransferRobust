import torch

from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Normalize
from src import settings
from src.config import set_seed
from src.utils import (logger, get_mean_and_std,
                        get_cifar_test_dataloader,
                        get_mnist_test_dataloader,
                        get_svhn_test_dataloader,
                        get_gtsrb_test_dataloder)
from autoattack import AutoAttack


EPSILON = 0.15

def make_eps(dataset: str) -> None:
    global EPSILON

    if dataset == "mnist":
        EPSILON = 0.15
    else:
        EPSILON = 8/255

    logger.info(f"using epsion: {EPSILON}")


SupportDatasetList = ['cifar10', 'cifar100', 'mnist', 'svhn', 'svhntl', 'gtsrb']
def get_test_dataset(dataset: str, batch_size=256) -> DataLoader:
    if dataset not in SupportDatasetList:
        raise ValueError("dataset not supported")
    if dataset.startswith("cifar"):
        return get_cifar_test_dataloader(dataset=dataset, normalize=False, shuffle=False, batch_size=batch_size)
    elif dataset == 'mnist':
        return get_mnist_test_dataloader(normalize=False, shuffle=False, batch_size=batch_size)
    elif dataset.startswith('svhn'):
        # 'svhn': using mean and std of 'svhn'
        # 'svhn': using mean and std of 'cifar100'
        return get_svhn_test_dataloader(dataset_norm_type=dataset, normalize=False, shuffle=False, batch_size=batch_size)
    elif dataset == "gtsrb":
        return get_gtsrb_test_dataloder(normalize=False, batch_size=batch_size)

class NormalizationWrapper(torch.nn.Module):
    def __init__(self, model, mean, std) -> None:
        super().__init__()
        self._model = model
        self.register_buffer("_mean", torch.tensor(mean).view(3, 1, 1))
        self.register_buffer("_std", torch.tensor(std).view(3, 1, 1))
    
    def forward(self, x):
        x = (x - self._mean) / self._std
        return self._model(x)


def accuracy(model, testset, device):
    items = 0
    acc = 0
    with torch.no_grad():
        for data, labels in testset:
            data = data.to(device) #type:torch.Tensor
            labels = labels.to(device) #type:torch.Tensor
            pred = model(data) #type:torch.Tensor
            items += labels.shape[0]
            acc += pred.argmax(dim=1).eq(labels).sum().item()
    
    logger.info(f"Accuracy: {acc/items}%")
    return acc/items

def auto_attack_robust(model, testset, device, log_path=None, batch_cnt=4):
    adversary = AutoAttack(model, norm="Linf", eps=EPSILON, log_path=log_path, version="standard", device=device)

    items = 0
    rob = 0
    with torch.no_grad():
        for it, (data, labels) in enumerate(testset):
            if it >= batch_cnt:
                break # auto-atk is too slow
            data = data.to(device) #type:torch.Tensor
            labels = labels.to(device) #type:torch.Tensor
            x_adv = adversary.run_standard_evaluation(data, labels, bs=data.shape[0])
            pred = model(x_adv) #type:torch.Tensor
            items += labels.shape[0]
            rob += pred.argmax(dim=1).eq(labels).sum().item()
    
    logger.info(f"Auto-Attack Robustness: {rob/items}%")
    return rob/items

def freeze_model_trainable_params(model:torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    
    logger.debug("all parameters are freezed")


def exp(model_path, args):
    set_seed(settings.seed)

    testset = get_test_dataset(args.dataset)

    model = get_model(args.model_type, args.num_classes, args.k).to(settings.device)
    model.load_state_dict(torch.load(model_path, map_location=settings.device))
    logger.debug(f"load from `{model_path}`")

    model.eval()
    freeze_model_trainable_params(model)

    mean, std = get_mean_and_std(args.dataset)
    model = NormalizationWrapper(model, mean, std).to(settings.device)

    acc = accuracy(model, testset, settings.device)
    
    
    atk_log_path = os.path.join(settings.log_dir, f"auto_atk_{os.path.basename(model_path)}.log")
    if not os.path.exists(atk_log_path):
        import pathlib 
        pathlib.Path(atk_log_path).touch()
    start_time = time.perf_counter()
    rob = auto_attack_robust(model, testset, settings.device, log_path=atk_log_path, batch_cnt=args.batch_cnt)
    end_time = time.perf_counter()
    logger.info(f"costing time: {end_time-start_time:.2f} secs")

    result = {
        model_path: {
                "Acc": acc,
                "Rob": rob
            }
        }

    logger.info(result)
    if args.result_file is not None:
        if not os.path.exists(os.path.dirname(args.result_file)):
            os.makedirs(os.path.dirname(args.result_file))
        
        if os.path.exists(args.result_file):
            with open(args.result_file, "r") as f:
                exist_data = json.load(f)
            for key in result.keys():
                exist_data[key] = result[key]
            result = exist_data
        
        with open(args.result_file, "w+") as f:
            json.dump(result, f)

if __name__ == "__main__":
    from src.cli.utils import get_model

    import time
    import json 
    import argparse
    import os

    parser  = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-n", "--num_classes", type=int, required=True)
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("-k", "--k", type=int, default=1)
    parser.add_argument("--log", type=str, default="auto_atk.log")
    parser.add_argument("--result-file", type=str, default=None)
    parser.add_argument("--batch-cnt", type=int, default=4)
    args = parser.parse_args()


    if args.model is None:
        model_list = [
            "trained_models/sntl_1_0.4_True_pres18_mnist_5_at_res18_svhn-best_robust-last",
            "trained_models/sntl_1_0.4_False_pres18_mnist_5_at_res18_svhn-best_robust-last",
            "trained_models/sntl_1_0.4_True_pres18_mnist_3_at_res18_svhn-best_robust-last",
            "trained_models/sntl_1_0.4_False_pres18_mnist_3_at_res18_svhn-best_robust-last",
            "trained_models/sntl_1_0.4_True_pres18_mnist_6_at_res18_svhn-best_robust-last",
            "trained_models/sntl_1_0.4_False_pres18_mnist_6_at_res18_svhn-best_robust-last",

            "trained_models/sntl_1_0.4_True_pres18_mnist_3_wd_fdm_True_res18_svhn_3_1.0-best_robust-last",
            "trained_models/sntl_1_0.4_False_pres18_mnist_3_wd_fdm_True_res18_svhn_3_1.0-best_robust-last",
            
            "trained_models/sntl_1_0.4_True_pres18_mnist_5_wd_fdm_True_res18_svhn_5_1.0-best_robust-last",
            "trained_models/sntl_1_0.4_False_pres18_mnist_5_wd_fdm_True_res18_svhn_5_1.0-best_robust-last",
            
            "trained_models/sntl_1_0.4_True_pres18_mnist_6_wd_fdm_True_res18_svhn_6_1.0-best_robust-last",
            "trained_models/sntl_1_0.4_False_pres18_mnist_6_wd_fdm_True_res18_svhn_6_1.0-best_robust-last"
        ]
    else:
        model_list = [args.model] 
    
    logger.change_log_file(settings.log_dir / args.log)
    make_eps(args.dataset)
    
    for model_path in model_list:
        exp(model_path, args)

    
    
    



    
