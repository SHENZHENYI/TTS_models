import torch
from torch import nn
from torch.optim import AdamW
import numpy as np
from transformers import get_cosine_schedule_with_warmup, \
                         get_cosine_with_hard_restarts_schedule_with_warmup, \
                         get_linear_schedule_with_warmup
                         
from typing import Dict, List, Tuple

def to_cuda(x: torch.Tensor, device: str) -> torch.Tensor:
    if device == 'cpu':
        return x
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.contiguous()
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
    return x

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class KeepAll:
    def __init__(self):
        self.all = []

    def add_batch(self, batch: torch.Tensor):
        """ the first dim is batch
        """
        for x in batch.detach().cpu():
            self.all.append(x)

def get_scheduler(scheduler: str, 
                  optimizer: object, 
                  num_train_steps: int, 
                  warmup_ratio: float, 
                  num_cycles: int
    ):
    if scheduler=='cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup_ratio * num_train_steps),
            num_training_steps=num_train_steps,
            num_cycles=0.5*num_cycles,
            last_epoch=-1,
        )
    elif scheduler=='cosine_hard':
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup_ratio * num_train_steps),
            num_training_steps=num_train_steps,
            num_cycles=num_cycles,
            last_epoch=-1,
        )
    elif scheduler=='linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup_ratio * num_train_steps),
            num_training_steps=num_train_steps,
            last_epoch=-1,
        )
    elif scheduler=='step_tacotron':
        gradual_learning_rates = (
            [0, 1e-1],
            [2e4, 5e-1],
            [4e5, 3e-1],
            [6e4, 1e-1],
            [8e4, 5e-2],
        )
        def stepwise_lr_change(step):
            last_lr = gradual_learning_rates[0][-1]
            for step_lr_set in gradual_learning_rates[1:]:
                step_thres, tar_lr = step_lr_set
                if step > step_thres:
                    last_lr = tar_lr
            return last_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, stepwise_lr_change)

    return scheduler

def mcrmse(targets: np.array, predictions: np.array) -> Tuple[np.float32, List[np.float32]]:
    rms_scores = np.sqrt(np.mean(np.square(targets - predictions), axis=0))
    return np.mean(rms_scores, axis=0)

def rmse_scores(targets: np.array, predictions: np.array) -> Tuple[np.float32, List[np.float32]]:
    rms_scores = np.sqrt(np.mean(np.square(targets - predictions), axis=0))
    return rms_scores

def count_parameters(model: nn.Module):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)