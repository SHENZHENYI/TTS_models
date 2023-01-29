import torch
from torch import nn
from torch.nn import functional as F
from librosa.filters import mel as librosa_mel_fn


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


class LinearBN(nn.Module):
    """Linear layer with Batch Normalization.

    x -> linear -> BN -> o

    Args:
        in_features (int): number of channels in the input tensor.
        out_features (int ): number of channels in the output tensor.
        bias (bool, optional): enable/disable bias in the linear layer. Defaults to True.
        init_gain (str, optional): method to set the gain for weight initialization. Defaults to 'linear'.
    """

    def __init__(self, in_features, out_features, bias=True, w_init_gain="linear"):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_features, out_features, bias=bias)
        self.batch_normalization = nn.BatchNorm1d(out_features, momentum=0.1, eps=1e-5)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """
        Shapes:
            x: [T, B, C] or [B, C]
        """
        out = self.linear_layer(x)
        if len(out.shape) == 3:
            out = out.permute(1, 2, 0)
        out = self.batch_normalization(out)
        if len(out.shape) == 3:
            out = out.permute(2, 0, 1)
        return out

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
        use_location_attention=True,
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


class LocationAwareAttention_v2(nn.Module):
    def __init__(
        self,
        attn_rnn_dim,
        attn_dim,
        encoder_dim,
        use_location_attention=True,
    ):
        super(LocationAwareAttention_v2, self).__init__()
        self.query_layer = Linear(attn_rnn_dim, attn_dim, bias=False, w_init_gain='tanh')
        self.source_layer = Linear(encoder_dim, attn_dim, bias=False, w_init_gain='tanh')
        self.v = Linear(attn_dim, 1, bias=True)
        if use_location_attention:
            self.location_layer = LocationLayer(attn_dim)
        self.mask_value = -float('inf')
        self.use_location_attention = use_location_attention
    
    def init_location_attention(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        self.attention_weights_cum = torch.zeros([B, T], device=inputs.device)

    def init_states(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        self.attention_weights = torch.zeros([B, T], device=inputs.device)
        if self.use_location_attention:
            self.init_location_attention(inputs)

    def preprocess_source_inputs(self, x):
        return self.source_layer(x)

    def update_location_attention(self, alignments):
        self.attention_weights_cum += alignments

    def get_alignment_energies(self, query, processed_source,):
        """
        ei,j = wT tanh(W si−1 + V hj + Ufi,j)
        Shapes:
            - query: (B, attn_rnn_dim)
            - processed_source: (B, encoder_len, attn_dim)
            - attn_weights_cat: (B, 2, encoder_len)
            - energies: (B, encoder_len)
        """
        attention_cat = torch.cat((self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)
        processed_query = self.query_layer(query.unsqueeze(1))

        if self.use_location_attention:
            processed_attention_cat= self.location_layer(attention_cat)
            energies = self.v(torch.tanh(processed_query + processed_source + processed_attention_cat))
        else:
            energies = self.v(torch.tanh(processed_query + processed_source))
        return energies.squeeze(2)


    def forward(self, query, source, processed_source, source_mask):
        """
        query: attention_rnn_hidden
        source: encoder_outputs

        Shapes:
            - attention_context: (B, encoder_dim)
            - attention_weights: (B, encoder_len)
        """
        energies = self.get_alignment_energies(
            query, processed_source,
        )

        energies = energies.data.masked_fill(~source_mask, self.mask_value) # added .data dont konw if will make a difference
        alignment = F.softmax(energies, dim=-1)

        if self.use_location_attention:
            self.update_location_attention(alignment)

        attention_context = torch.bmm(alignment.unsqueeze(1), source)
        attention_context = attention_context.squeeze(1)
        self.attention_weights = alignment

        return attention_context
