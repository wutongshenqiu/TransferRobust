"""
Record inputs for last k blocks
"""
import torch
import numpy as np
import os
import contextlib

from torch import Tensor
from torch.nn import Module
from src.utils import logger
from src.cli.utils import get_train_dataset, get_test_dataset
from src.networks import make_blocks, wrn34_10, parseval_retrain_wrn34_10
from src.config import settings, set_seed
from src.attack import LinfPGDAttack

from typing import Tuple, List, Dict


class ForwardHook(object):
    def __init__(self, module:Module):
        super(ForwardHook).__init__()

        self._module_input = [None]
        self._module_output = None
        
        self.module = module
        self._module_handle = None
    
    @property
    def module_input(self)->Tensor:
        return self._module_input
    
    @property
    def module_output(self)->List[Tensor]:
        return self._module_output

    def hook(self, fn=None):
        def _hook_fn(module:Module, input:Tuple[Tensor], output:Tensor):
            if fn is not None:
                output = fn(module, input, output)
            # based on the document of PyTorch, the input is a tuple
            # of Tensor (may be for multiple inputs function)
            # https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks
            self._module_input = [i.clone().detach() for i in input]
            # different from input, output is a tensor
            self._module_output = output.clone().detach()
            return output

        if self._module_handle is not None:
            self.remove()
        self._module_handle = self.module.register_forward_hook(_hook_fn)

        return self
            
    def remove(self):
        if self._module_handle is not None:
            self._module_handle.remove()
            self._module_handle = None
        
        self._module_input = [None]
        self._module_output = None


@contextlib.contextmanager
def register_forward_hook_to_k_block(model, k):
    blocks = make_blocks(model)
    total_blocks = blocks.get_total_blocks()

    wanted_block_name = f"block{total_blocks - k + 1}"
    block = getattr(blocks, wanted_block_name)
    if isinstance(block, torch.nn.Sequential):
        block = block[0]
    logger.info(f"get {wanted_block_name}")

    fh = ForwardHook(block)
    fh.hook()
    yield fh
    fh.remove()

@contextlib.contextmanager 
def hook_last_k_blocks(model, k):
    blocks = make_blocks(model)
    total_blocks = blocks.get_total_blocks()

    fhs = {} #type:Dict[int, ForwardHook]
    for _k in range(1, k+1):
        wanted_block_name = f"block{total_blocks - _k + 1}"
        block = getattr(blocks, wanted_block_name)
        if isinstance(block, torch.nn.Sequential):
            block = block[0]
        logger.info(f"get {wanted_block_name}")
        
        fh = ForwardHook(block)
        fh.hook()
        fhs[_k] = fh

    yield fhs

    for fh in fhs.values():
        fh.remove()


def get_model(model_type, output_label_num, k=None):
    if model_type == "wrn34":
        model = wrn34_10(num_classes=output_label_num)
    elif model_type == "pwrn34":
        if k == None:
            raise ValueError(f"please provide a valid 'k'")
        model = parseval_retrain_wrn34_10(num_classes=output_label_num, k=k)
    else:
        raise ValueError(f"Not Support Model Type '{model_type}'")

    logger.warning(f"YOU ARE USING {type(model).__name__}")

    return model


LOG_FILE = "fd.log"
DEVICE = "cuda"

FREEZE_K = 8

def exp(k, model_path=None, ds="cifar10", ds_type="train", model_type="wrn34", trainset="cifar100"):
    def yield_last_k_range(_k=None):
        if _k is not None: yield _k
        for _k in range(FREEZE_K, 0, -1):
            yield _k

    logger.change_log_file(settings.log_dir / LOG_FILE)

    if ds_type == "train":
        dataloader = get_train_dataset(ds)
    else:
        dataloader = get_test_dataset(ds)

    if trainset == "cifar100":
        output_label_num = 100
    else:
        output_label_num = 10

    attack_params = {
        "random_init": 1,
        "epsilon": 8/255,
        "step_size": 2/255,
        "num_steps": 20,
        "dataset_name": "cifar100",
    }

    if model_path is None:
        # model_name_list = [
        #     "sntl_1_0.6_True_pwrn34_cifar10_4_cartl_wrn34_cifar100_4_0.01-best_robust-last",
        #     "sntl_1_0.6_True_pwrn34_cifar10_6_cartl_wrn34_cifar100_6_0.01-best_robust-last",
        #     "sntl_1_0.6_True_pwrn34_cifar10_8_cartl_wrn34_cifar100_8_0.01-best_robust-last",

        #     "sntl_1_0.6_False_pwrn34_cifar10_4_at_wrn34_cifar100-best_robust-last",
        #     "sntl_1_0.6_False_pwrn34_cifar10_6_at_wrn34_cifar100-best_robust-last",
        #     "sntl_1_0.6_False_pwrn34_cifar10_8_at_wrn34_cifar100-best_robust-last",

        #     "sntl_1_0.6_pwrn34_cifar10_4_cartl_wrn34_cifar100_4_0.01-best_robust-last",
        #     "sntl_1_0.6_pwrn34_cifar10_6_cartl_wrn34_cifar100_6_0.01-best_robust-last",
        #     "sntl_1_0.6_pwrn34_cifar10_8_cartl_wrn34_cifar100_8_0.01-best_robust-last",
        # ]
        model_name_list = [
            f"sntl_1_0.6_True_pwrn34_cifar10_{k}_cartl_wrn34_cifar100_{k}_0.01-best_robust-last",

            f"sntl_1_0.6_False_pwrn34_cifar10_{k}_at_wrn34_cifar100-best_robust-last",

            f"sntl_1_0.6_pwrn34_cifar10_{k}_cartl_wrn34_cifar100_{k}_0.01-best_robust-last",
        ]
    else:
        model_name_list = [model_path]
    
    for model_name in model_name_list:
        logger.info(f"using model {model_name}")
        set_seed(settings.seed)
        
        model = get_model(model_type, output_label_num, k).to(DEVICE)
        mp = os.path.join(settings.model_dir, model_name)
        model.load_state_dict(state_dict = torch.load(mp, map_location=DEVICE))
        model.eval()

        attacker = LinfPGDAttack(model=model, **attack_params)

        save_dir = f"misc_results/all_l2norm/{ds}_{ds_type}/{model_name}_last{FREEZE_K}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        items = 0

        acc = 0
        rob = 0

        k_clean_features:Dict[int, Tensor] = { }
        k_adv_features:Dict[int, Tensor] = { }

        k_clean_features_sum:Dict[int, Tensor] = { key:torch.tensor(0) for key in yield_last_k_range() }
        k_adv_features_sum:Dict[int, Tensor] = { key:torch.tensor(0) for key in yield_last_k_range() }

        k_norms:Dict[int, List[np.ndarray]] = { key:[] for key in yield_last_k_range() }

        with hook_last_k_blocks(model, FREEZE_K) as fhs:
            for data, labels in dataloader:
                data = data.to(DEVICE)
                labels = labels.to(DEVICE)

                items += labels.shape[0]

                with torch.no_grad():
                    pred = model(data) # type:Tensor

                acc += pred.argmax(dim=1).eq(labels).sum().item()

                for _k in yield_last_k_range():
                    assert _k in fhs
                    assert _k in k_clean_features_sum
                    k_clean_features[_k] = fhs[_k].module_input[0]
                    k_clean_features_sum[_k] = k_clean_features[_k].sum(dim=0) + k_clean_features_sum[_k]
                    

                adv_data = attacker.calc_perturbation(data, labels)

                with torch.no_grad():
                    pred = model(adv_data)

                rob += pred.argmax(dim=1).eq(labels).sum().item()
                
                for _k in yield_last_k_range():
                    assert _k in fhs
                    assert _k in k_adv_features_sum
                    k_adv_features[_k] = fhs[_k].module_input[0]
                    k_adv_features_sum[_k] = k_adv_features[_k].sum(dim=0) + k_adv_features_sum[_k]

                for _k in yield_last_k_range():
                    scalar = torch.norm(
                        (k_adv_features[_k] - k_clean_features[_k]).view(k_clean_features[_k].shape[0], -1),
                        p=2,
                        dim=1
                    ) #type:Tensor

                    k_norms[_k].append(scalar.detach().cpu().numpy())
        
        for _k in yield_last_k_range(): 
            k_norms[_k] = np.concatenate(k_norms[_k])
        
        import pickle
        import json

        with open(os.path.join(save_dir, "meta.json"), "w+") as f:
            json.dump({
                "items": items,
                "acc": acc/items,
                "rob": rob/items
            }, f)

        with open(os.path.join(save_dir, "result.pkl"), "wb+") as f:
            pickle.dump(k_norms, f)

        for _k in yield_last_k_range():
            k_clean_features_sum[_k] = (k_clean_features_sum[_k] / items).detach().cpu().numpy()
            k_adv_features_sum[_k] = (k_adv_features_sum[_k] / items).detach().cpu().numpy()
        
        with open(os.path.join(save_dir, "clean_mean.pkl"), "wb+") as f:
            pickle.dump(k_clean_features_sum, f)
        
        with open(os.path.join(save_dir, "adv_mean.pkl"), "wb+") as f:
            pickle.dump(k_adv_features_sum, f)
        
        


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--k", type=int, required=True)
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("--log", type=str, default=LOG_FILE)
    parser.add_argument("--model-type", type=str, default="wrn34")
    parser.add_argument("--trainset", type=str, default="cifar100")
    parser.add_argument("--freeze-k", type=int, default=FREEZE_K)

    args = parser.parse_args()

    LOG_FILE = args.log
    FREEZE_K = args.freeze_k

    exp(k=args.k, model_path=args.model, ds="cifar10", ds_type="test", model_type=args.model_type, trainset=args.trainset)
