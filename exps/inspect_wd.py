from exps.diagnose_network import LOG_FILE
import torch
import numpy as np
import os
import contextlib
import json
import pickle

from torch import Tensor
from torch.nn import Module
from src.utils import logger
from src.cli.utils import get_train_dataset, get_test_dataset, get_model
from src.networks import make_blocks, wrn34_10, parseval_retrain_wrn34_10
from src.config import settings, set_seed
from src.attack import LinfPGDAttack

from src.trainer.robust_plus_regularization_trainer.robust_plus_wasserstein_trainer import _Estimator

from typing import Tuple, List, Dict, Union


class ForwardHook(object):
    def __init__(self, module:Module, func=None):
        super(ForwardHook).__init__()

        self._module_input:List[Tensor] = []
        self._module_output:Tensor = None
        
        self.module = module
        self._module_handle = None
        self._func = func
    
    @property
    def want(self):
        if self._func is None:
            return self.module_output
        return self._func(self)
    
    @property
    def module_input(self)->List[Tensor]:
        return self._module_input
    
    @property
    def module_output(self)->Tensor:
        assert self._module_output is not None
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
        
        self._module_input = []
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

    fh = ForwardHook(block, func=lambda h:h.module_input[0])
    fh.hook()
    yield fh
    fh.remove()

def freeze_model_trainable_params(model:Module):
    for param in model.parameters():
        param.requires_grad = False
    
    logger.debug("all parameters are freezed")

def prepare_estimator(model, loader, hook:ForwardHook, estimator_path:str)->_Estimator:
        # first, we get an input for calculating dimension of features
        image, _ = next(iter(loader))
        image = image.to(settings.device)

        with torch.no_grad():
            model(image)
        
        dim = torch.prod(torch.tensor(hook.want.shape[1:]))
        estimator = _Estimator(dim).to(settings.device)

        assert estimator_path.startswith("checkpoint") or estimator_path.startswith("./checkpoint")
        obj = torch.load(settings.checkpoint_dir/estimator_path, map_location=settings.device)
        estimator.load_state_dict(obj["estimator_weights"])

        return estimator 

def exp(k, model_path, model_type, ds, ds_type, num_classes, estimator):
    
    logger.change_log_file(settings.log_dir / LOG_FILE)
    if ds_type == "train":
        dataloader = get_train_dataset(ds)
    else:
        dataloader = get_test_dataset(ds)
    
    attack_params = {
        "random_init": 1,
        "epsilon": 8/255,
        "step_size": 2/255,
        "num_steps": 20,
        "dataset_name": ds,
    }
    
    if model_path is None:
        models_list = [

        ]
    else:
        models_list = [model_path]

    for model_name in models_list:
        logger.info(f"using model {model_name}")
        set_seed(settings.seed)

        model = get_model(model_type, num_classes, k)
        model.load_state_dict(torch.load(os.path.join(settings.model_dir, model_name), map_location=settings.device))
        model.eval()
        freeze_model_trainable_params(model)

        attacker = LinfPGDAttack(model=model, **attack_params)

        save_dir = f"misc_results/wd/{ds}_{ds_type}/{model_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        items = 0
        acc = 0
        rob = 0

        clean_we:List[np.ndarray] = []
        adv_we:List[np.ndarray] = []

        with register_forward_hook_to_k_block(model, k) as fh:
            estimator = prepare_estimator(model, dataloader, fh, estimator)

            for data, labels in dataloader:
                data:Tensor = data.to(settings.device)
                labels:Tensor = data.to(settings.device)

                items += labels.shape[0]

                with torch.no_grad():
                    model(data) #type:Tensor
                    we = estimator(fh.want) #type:Tensor
                clean_we.append(we.detach().cpu().numpy())

                adv_data = attacker.calc_perturbation(data, labels)
                with torch.no_grad():
                    model(adv_data)
                    we = estimator(fh.want)
                adv_we.append(we.detach().cpu().numpy())
            
        
        clean_we = np.concatenate(clean_we, axis=0)
        adv_we = np.concatenate(adv_we, axis=0)

        with open(os.path.join(save_dir, "clean_we.pkl"), "wb+") as f:
            pickle.dump(clean_we, f)
        
        with open(os.path.join(save_dir, "adv_we.pkl"), "wb+") as f:
            pickle.dump(adv_we, f)
        

LOG_FILE = "wd.log"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--k", type=int, required=True)
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("--log", type=str, default=LOG_FILE)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-type", type=str, required="test")
    parser.add_argument("--model-type", type=str, default="wrn34")
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--estimator", type=str, required=True)

    args = parser.parse_args()

    LOG_FILE = args.log

    exp(
        k=args.k,
        model_path=args.model, 
        model_type=args.model_type, 
        ds=args.dataset, 
        ds_type=args.dataset_type, 
        num_classes=args.num_classes, 
        estimator=args.estimator
    )





