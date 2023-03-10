from abc import abstractmethod
from typing import Dict, List, Tuple
import torch
from torch import nn

from src.utils.utils import sequence_mask

class BaseModel(nn.Module):
    """abstract base model"""
    @abstractmethod
    def forward(self, input: torch.Tensor, *args, **kwargs) -> Dict:
        """forward
        """
        outputs = {}
        return outputs

    @abstractmethod
    def train_step(self, batch: Dict, criterion: nn.Module) -> Tuple[Dict, torch.Tensor]:
        """do one training step
        """
        outputs = {}
        losses = {}
        return outputs, losses

    @abstractmethod
    def eval_step(self, batch: Dict, criterion: nn.Module) -> Tuple[Dict, Dict]:
        """do one evaluation step
        """
        outputs = {}
        losses = {}
        return outputs, losses

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """load checkpoint for evaluation
        """

    @staticmethod
    def compute_masks(lengths):
        """Compute masks against sequence paddings."""
        # B x T_in_max (boolean)
        masks = sequence_mask(lengths)
        return masks