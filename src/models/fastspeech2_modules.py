import torch
from torch import nn
from typing import List

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
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        conv_hidden_size: int,
        kernel_size: List[int],
        dropout: int = 0.1,
    ):
        super(FFTBlock, self).__init__()
        self.slf_attn = nn.MultiheadAttention(input_dim, n_heads, dropout=dropout)
        self.ln_1 = LayerNorm(input_dim)
        self.conv1ds = nn.Sequential(nn.OrderedDict([
            ('conv1d1', nn.Conv1d(input_dim, conv_hidden_size, kernel_size=kernel_size[0], padding=(kernel_size[0] - 1)//2)),
            ('relu', nn.ReLU()),
            ('conv1d2', nn.Conv1d(conv_hidden_size, input_dim, kernel_size=kernel_size[1], padding=(kernel_size[1] - 1)//2)),
        ]))
        self.ln_2 = LayerNorm(input_dim)

    def forward(self, x, encoder_mask):
        # x is the encoder inputs
        x_norm = self.ln_1(x)
        x = x + self.slf_attn(x_norm, x_norm, x_norm, attn_mask=encoder_mask)
        x = x + self.conv1ds(self.ln_2(x))
        return x


class Encoder(nn.Module):
    """ encoder of FastSpeech2
    consists of a phoneme embedding, a positional embedding, a fft block
    """
    def __init__(
        self,
        n_layers,
        d_model,
        n_head,
        conv_hidden_size,
        kernel_size,
    ):
        super(Encoder, self).__init__()
        self.n_layers = n_layers

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.ffts = nn.ModuleList([
            FFTBlock(d_model, n_head, conv_hidden_size, kernel_size) for _ in range(n_layers)
        ])

    def forward(
        self, x, src_mask
    ):
        for fft in self.ffts:
            x = fft(x, src_mask)
        return x
    

class VarianceAdaptor():
    def __init__(self):
        pass 

class Decoder():
    """Mel-spectrogram Decoder"""
    def __init__(self):
        pass