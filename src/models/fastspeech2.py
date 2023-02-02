"""
Implementation of https://arxiv.org/pdf/2006.04558.pdf 
"""
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Dict, Tuple, List

from hparams import hparams as hps
from src.models.basemodel import BaseModel
from src.models.fastspeech2_modules import (
    FeedForwardTransformer,
    VarianceAdaptor
)
from src.dataset.dataset import TTSDataset
from src.utils.utils import sequence_mask
from src.trainer_utils import get_scheduler

class FastSpeech2(BaseModel):
    def __init__(
        self,
        preproc_cfg,
        model_cfg
        
    ):
        super(FastSpeech2, self).__init__()
        self.max_mel_len = model_cfg.max_seq_len
        self.src_word_emb = nn.Embedding(
            preproc_cfg.phoneme_vocab_size, model_cfg.encoder_hidden, padding_idx=0
        )
        self.encoder = FeedForwardTransformer(
            max_seq_len=model_cfg.max_seq_len,
            n_layers=model_cfg.encoder_layer,
            encoder_hidden_dim=model_cfg.encoder_hidden,
            n_head=model_cfg.encoder_head,
            conv_kernel_size=model_cfg.encoder_conv_kernel_size,
            conv_filter_size=model_cfg.encoder_conv_filter_size,
            dropout=model_cfg.encoder_dropout
        )
        self.variance_adaptor = VarianceAdaptor(
            input_dim=model_cfg.encoder_hidden,
            conv_filter_size=model_cfg.variance_predictor_conv_filter_size,
            conv_kernel_size=model_cfg.variance_predictor_conv_kernel_size,
            dropout=model_cfg.variance_predictor_dropout,
            pitch_scale=model_cfg.variance_predictor_pitch_scale,
            energy_scale=model_cfg.variance_predictor_energy_scale,
            n_bins=model_cfg.n_bins,
            pitch_energy_stats_path=os.path.join(preproc_cfg.preprocessed_path, 'stats.json'),
        )
        self.decoder = FeedForwardTransformer(
            max_seq_len=model_cfg.max_seq_len,
            n_layers=model_cfg.decoder_layer,
            encoder_hidden_dim=model_cfg.decoder_hidden,
            n_head=model_cfg.decoder_head,
            conv_kernel_size=model_cfg.decoder_conv_kernel_size,
            conv_filter_size=model_cfg.decoder_conv_filter_size,
            dropout=model_cfg.decoder_dropout
        )
        self.mel_linear = nn.Linear(model_cfg.decoder_hidden, preproc_cfg.n_mel_channels)
    
    def forward(
        self, text, text_lengths, mel_lengths, \
        pitch_targets=None, energy_targets=None, duration_targets=None, \
    ):        
        # compute masks
        text_masks = self.compute_masks(text_lengths)
        mel_masks = self.compute_masks(mel_lengths)

        # embedding
        text = self.src_word_emb(text)

        # encoder side
        encoder_outputs = self.encoder(text, text_masks)

        # variance adaptor
        output, log_duration_preds, pitch_preds, energy_preds, mel_len, mel_masks = self.variance_adaptor(
            encoder_outputs, text_masks, self.max_mel_len, mel_masks, pitch_targets, energy_targets, duration_targets,
        )

        # decoder side
        output = self.decoder(output, mel_masks)

        # to mel shape
        output = self.mel_linear(output)

        return {
            'model_outputs': output,
            'pitch_outputs': pitch_preds,
            'energy_outputs': energy_preds,
            'duration_outputs': log_duration_preds,
            'mel_masks': mel_masks
        }

