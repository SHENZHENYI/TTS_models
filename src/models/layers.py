import torch
from torch import nn
from torch.nn import functional as F

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
    ):
        super(LocationAwareAttention, self).__init__()
        self.query_layer = Linear(attn_rnn_dim, attn_dim, bias=False, w_init_gain='tanh')
        self.source_layer = Linear(encoder_dim, attn_dim, bias=False, w_init_gain='tanh')
        self.v = Linear(attn_dim, 1, bias=True)
        self.location_layer = LocationLayer(attn_dim)
        self.mask_value = -float('inf')
    
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
        processed_attn_weights = self.location_layer(attn_weights_cat)
        energies = self.v(torch.tanh(processed_query + processed_source + processed_attn_weights))
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


