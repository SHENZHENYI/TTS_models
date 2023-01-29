"""
Implementation of https://arxiv.org/pdf/2006.04558.pdf 
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Dict, Tuple, List

from hparams import hparams as hps
from src.models.basemodel import BaseModel
from src.models.fastspeech2_modules import *
from src.dataset.dataset import TTSDataset
from src.utils.utils import sequence_mask
from src.trainer_utils import get_scheduler

class FastSpeech2(BaseModel):
    def __init__(
        self,
        hps
    ):
        super(FastSpeech2, self).__init__()
        self.encoder = Encoder()
        self.variance_adaptor = VarianceAdaptor()
        self.decoder = Decoder()
    
    def forward(
        self, text, text_lengths, mel, mel_lengths=None, \
        pitch_targets=None, energy_targets=None, duration_targets=None, \
        pitch_control=None, energy_control=None, duration_control=None, 
    ):        
        # compute masks
        text_masks = self.compute_masks(text_lengths)
        mel_masks = self.compute_masks(mel_lengths)

        # encoder side
        encoder_outputs = self.encoder(text, text_lengths)

        # variance adaptor
        output, pitch_preds, energy_preds, duration_preds = self.variance_adaptor(
            encoder_outputs, text_masks, pitch_targets, energy_targets, duration_targets,
            pitch_control, energy_control, duration_control
        )

        # decoder side
        output = self.decoder(output, mel_masks)

        return {
            'model_outputs': output,
            'pitch_outputs': pitch_preds,
            'energy_outputs': energy_preds,
            'duration_outputs': pitch_preds
        }

