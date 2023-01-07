import torch
from torch import nn
from torch.nn import functional as F
from librosa.filters import mel as librosa_mel_fn

from src.utils.audio_processing import dynamic_range_compression, dynamic_range_decompression
from src.utils.stft import STFT

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel_size,
        out_channel_size=None,
        kernel_size=5,
        activation='relu'
    ):
        super(ConvBlock, self).__init__()
        if out_channel_size is None:
            out_channel_size = in_channel_size
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channel_size, out_channel_size, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channel_size, momentum=0.1, eps=1e-5)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
        self.dropout = nn.Dropout(p=0.5)
    
        #if activation is not None:
        #    nn.init.xavier_uniform_(
        #        self.conv.weight,
        #        gain=nn.init.calculate_gain(activation))

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        if self.dropout is not None:
            return self.dropout(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class LocationLayer(nn.Module):
    """Layers for Location Sensitive Attention
    Args:
        attention_dim (int): number of channels in the input tensor.
        attention_n_filters (int, optional): number of filters in convolution. Defaults to 32.
        attention_kernel_size (int, optional): kernel size of convolution filter. Defaults to 31.
    """

    def __init__(self, attention_dim, attention_n_filters=32, attention_kernel_size=31):
        super().__init__()
        self.location_conv1d = nn.Conv1d(
            in_channels=2,
            out_channels=attention_n_filters,
            kernel_size=attention_kernel_size,
            stride=1,
            padding=(attention_kernel_size - 1) // 2,
            bias=False,
        )
        self.location_dense = Linear(attention_n_filters, attention_dim, bias=False, w_init_gain="tanh")

    def forward(self, attention_cat):
        """
        Shapes:
            attention_cat: [B, 2, C]
        """
        processed_attention = self.location_conv1d(attention_cat)
        processed_attention = self.location_dense(processed_attention.transpose(1, 2))
        return processed_attention

class LocationAwareAttention(nn.Module):
    def __init__(
        self,
        attn_rnn_dim,
        attn_dim,
        encoder_dim,
        use_location_attention=False,
    ):
        super(LocationAwareAttention, self).__init__()
        self.query_layer = Linear(attn_rnn_dim, attn_dim, bias=False, w_init_gain='tanh')
        self.source_layer = Linear(encoder_dim, attn_dim, bias=False, w_init_gain='tanh')
        self.v = Linear(attn_dim, 1, bias=True)
        if use_location_attention:
            self.location_layer = LocationLayer(attn_dim)
        self.mask_value = -float('inf')
        self.use_location_attention = use_location_attention
    
    def preprocess_source_inputs(self, x):
        return self.source_layer(x)

    def get_alignment_energies(self, query, processed_source, attn_weights_cat):
        """
        ei,j = wT tanh(W si−1 + V hj + Ufi,j)
        Shapes:
            - query: (B, attn_rnn_dim)
            - processed_source: (B, encoder_len, attn_dim)
            - attn_weights_cat: (B, 2, encoder_len)
            - energies: (B, encoder_len)
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        if self.use_location_attention:
            processed_attn_weights = self.location_layer(attn_weights_cat)
            energies = self.v(torch.tanh(processed_query + processed_source + processed_attn_weights))
        else:
            energies = self.v(torch.tanh(processed_query + processed_source))
        return energies.squeeze(2)


    def forward(self, query, source, processed_source, attn_weights_cat, source_mask):
        """
        query: attention_rnn_hidden
        source: encoder_outputs

        Shapes:
            - attention_context: (B, encoder_dim)
            - attention_weights: (B, encoder_len)
        """
        alignment = self.get_alignment_energies(
            query, processed_source, attn_weights_cat
        )
        alignment = alignment.masked_fill(~source_mask, self.mask_value)
        attention_weights = F.softmax(alignment, dim=-1)

        attention_context = torch.bmm(attention_weights.unsqueeze(1), source)

        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights

class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax
        )

        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output