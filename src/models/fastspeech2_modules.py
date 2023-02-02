import os
import json
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List
from collections import OrderedDict

from src.utils.utils import sequence_mask

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class FFTBlock(nn.Module):
    """Feedforward Transformer block
    the same as the standard transformer block but using a conv1d layer instead of a mlp
    """
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        conv_hidden_size: int,
        kernel_size: List[int],
        dropout: int = 0.1,
    ):
        super(FFTBlock, self).__init__()
        self.slf_attn = nn.MultiheadAttention(input_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln_1 = LayerNorm(input_dim)
        self.conv1ds = nn.Sequential(OrderedDict([
            ('conv1d1', nn.Conv1d(input_dim, conv_hidden_size, kernel_size=kernel_size[0], padding=(kernel_size[0] - 1)//2)),
            ('relu', nn.ReLU()),
            ('conv1d2', nn.Conv1d(conv_hidden_size, input_dim, kernel_size=kernel_size[1], padding=(kernel_size[1] - 1)//2)),
        ]))
        self.ln_2 = LayerNorm(input_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, encoder_mask):
        # x is the encoder inputs
        x = self.ln_1(x + self.slf_attn(x, x, x, key_padding_mask=encoder_mask)[0]) # postnorm
        x = self.ln_2(x + self.dropout(self.conv1ds(x.transpose(1,2)).transpose(1,2))) # transpose for conv
        return x

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

class LengthRegulator(nn.Module):
    def __init__(
        self
    ):
        super(LengthRegulator, self).__init__()

    def forward(
        self, x, duration, max_mel_len
    ):
        """expand x_i dur_i times, where i the the index of the two tensors"""
        B, T, C = x.shape
        outputs = []
        expaned_mel_len = []
        for b in range(B):
            expand_matrix = torch.zeros(max_mel_len, T, device=x.device)
            accum = 0
            for t in range(T):
                expand_matrix[accum:accum+duration[b, t], t] = 1.
                accum += duration[b, t]
            expaned_mel_len.append(accum)
            outputs.append(expand_matrix @ x[b])
        return pad(outputs), torch.tensor(expaned_mel_len, device=x.device)


class VariancePredictor(nn.Module):
    """Variance Predictor
    predicts variance like duration, energy, pitch"""
    def __init__(
        self,
        input_dim,
        conv_filter_size,
        conv_kernel_size,
        dropout,
    ):
        super(VariancePredictor, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_filter_size, kernel_size=conv_kernel_size, padding=(conv_kernel_size - 1)//2)
        self.ln1 = LayerNorm(conv_filter_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(conv_filter_size, conv_filter_size, kernel_size=conv_kernel_size, padding=(conv_kernel_size - 1)//2)
        self.ln2 = LayerNorm(conv_filter_size)
        self.dropout2 = nn.Dropout(p=dropout)
        self.linear = nn.Linear(conv_filter_size, 1)
    
    def forward(self, x, src_mask=None):
        x = self.dropout1(self.ln1(self.conv1(x.contiguous().transpose(1, 2)).contiguous().transpose(1, 2).relu()))
        x = self.dropout2(self.ln2(self.conv2(x.contiguous().transpose(1, 2)).contiguous().transpose(1, 2).relu()))
        x = self.linear(x)
        x = x.squeeze(-1)

        if src_mask is not None:
            x = x.masked_fill(src_mask==0, 0.)

        return x


class FeedForwardTransformer(nn.Module):
    """ encoder of FastSpeech2
    consists of a phoneme embedding, a positional embedding, a fft block
    """
    def __init__(
        self,
        max_seq_len,
        n_layers,
        encoder_hidden_dim,
        n_head,
        conv_filter_size,
        conv_kernel_size,
        dropout,
    ):
        super(FeedForwardTransformer, self).__init__()
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(max_seq_len+1, encoder_hidden_dim).unsqueeze(0),
            requires_grad=False,
        )

        self.ffts = nn.ModuleList([
            FFTBlock(encoder_hidden_dim, n_head, conv_filter_size, conv_kernel_size, dropout) for _ in range(n_layers)
        ])

    def forward(
        self, x, src_mask
    ):
        B, T, C = x.shape
        max_len = min(T, self.max_seq_len)
        src_mask = src_mask[:, :max_len]

        x = x[:, :max_len, :] + self.position_enc[:, :max_len, :]
        for fft in self.ffts:
            x = fft(x, ~src_mask)
        return x
    

class VarianceAdaptor(nn.Module):
    """Variance Adaptor
    A stack of variance predictors, and a duration regulator"""
    def __init__(
        self,
        input_dim,
        conv_filter_size,
        conv_kernel_size,
        dropout,
        pitch_scale,
        energy_scale,
        n_bins,
        pitch_energy_stats_path,
    ):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(input_dim, conv_filter_size, conv_kernel_size, dropout)
        self.pitch_predictor = VariancePredictor(input_dim, conv_filter_size, conv_kernel_size, dropout)
        self.energy_predictor = VariancePredictor(input_dim, conv_filter_size, conv_kernel_size, dropout)
        self.length_regulator = LengthRegulator()

        assert pitch_scale in ['log', 'linear']
        assert energy_scale in ['log', 'linear']

        with open(
            os.path.join(pitch_energy_stats_path)
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        self.init_bins("pitch_bins", pitch_min, pitch_max, n_bins, pitch_scale)
        self.init_bins("energy_bins", energy_min, energy_max, n_bins, energy_scale)

        self.pitch_embeddings = nn.Embedding(
            n_bins, input_dim, 
        )
        self.energy_embeddings = nn.Embedding(
            n_bins, input_dim, 
        )

    def init_bins(self, name, min_val, max_val, n_bins, scale):
        if scale == 'log':
            self.register_buffer(
                name,
                torch.exp(torch.linspace(np.log(min_val+1e-10), np.log(max_val), n_bins-1))
            )
        else:
            self.register_buffer(
                name,
                torch.linspace(min_val+1e-10, max_val, n_bins-1)
            )
    
    def get_variance_embeddings_train(self, predictor, embedding_layer, x, src_mask, target, bins):
        prediction = predictor(x, src_mask)
        embedding = embedding_layer(torch.bucketize(target, bins))
        return prediction, embedding

    def get_variance_embeddings_inference(self, predictor, embedding_layer, x, src_mask, target, bins, control):
        prediction = predictor(x, src_mask)
        prediction = prediction * control
        embedding = embedding_layer(torch.bucketize(prediction, bins))
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        max_mel_len,
        mel_mask=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None
    ):
        """for train"""
        max_mel_len = min(max_mel_len, mel_mask.size(1))

        # duration
        log_duration_prediction = self.duration_predictor(x, src_mask)
        x, mel_len = self.length_regulator(x, duration_target, max_mel_len)

        # pitch
        pitch_prediction, pitch_embedding = self.get_variance_embeddings_train(
                self.pitch_predictor, self.pitch_embeddings, x, mel_mask, pitch_target, self.pitch_bins
            )
        x = x + pitch_embedding

        # energy
        energy_prediction, energy_embedding = self.get_variance_embeddings_train(
                self.energy_predictor, self.energy_embeddings, x, mel_mask, energy_target, self.energy_bins
            )
        x = x + energy_embedding

        return x, log_duration_prediction, pitch_prediction, energy_prediction, mel_len, mel_mask
    
    def inference(
        self,
        x,
        src_mask,
    ):
        """inference forward"""
        pass
