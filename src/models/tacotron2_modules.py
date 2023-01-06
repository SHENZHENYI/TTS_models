import torch
from torch import nn
from torch.nn import functional as F
from typing import List

from src.models.layers import ConvBlock, Linear, LocationAwareAttention

class Encoder(nn.Module):
    def __init__(
        self,
        cnn_channels,
        cnn_kernel_size,
        num_cnns,
        lstm_hidden_size,
    ):
        super(Encoder, self).__init__()
        self.conv_blocks = nn.ModuleList([ConvBlock(cnn_channels, cnn_channels, cnn_kernel_size) for _ in range(num_cnns)])
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden_size//2, \
                        num_layers=1, batch_first=True, bias=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.conv_blocks:
            x = conv(x)
        x = x.transpose(1, 2)
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        self.lstm.flatten_parameters()
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x

class PreNet(nn.Module):
    def __init__(
        self,
        in_size: int,
        sizes: List[int],
        dropout_at_inference=False,
    ):
        super(PreNet, self).__init__()
        self.dropout_at_inference = dropout_at_inference
        in_sizes = [in_size] + sizes[:-1]
        self.layers = nn.ModuleList([
            Linear(in_size, out_size) for in_size, out_size in zip(in_sizes, sizes)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = F.dropout(layer(x).relu(), p=0.5, training=self.training or self.dropout_at_inference)
        return x

class PostNet(nn.Module):
    def __init__(
        self,
        channels,
        n_convs
    ):
        super(PostNet, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(ConvBlock(channels, 512, kernel_size=5, activation='tanh'))
        for _ in range(1, n_convs-1):
            self.convs.append(ConvBlock(512, 512, kernel_size=5, activation='tanh'))
        self.convs.append(ConvBlock(512, channels, kernel_size=5, activation=None))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        encoder_dim,
        n_frames_per_step,
        frame_dim,
        attn_rnn_dim = 1024,
        decoder_rnn_dim = 1024,
        prenet_dim = 256,
        attn_dim = 128,
        p_attn_dropout = 0.1,
        p_decoder_dropout = 0.1,
        max_decoder_steps = 1000,
        stop_threshold = 0.99999
    ):
        super(Decoder, self).__init__()
        self.max_decoder_steps = max_decoder_steps
        self.stop_threshold = stop_threshold
        self.n_frames_per_step = n_frames_per_step
        self.frame_dim = frame_dim
        self.encoder_dim = encoder_dim
        self.attn_rnn_dim = attn_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.p_attn_dropout = p_attn_dropout
        self.p_decoder_dropout = p_decoder_dropout

        self.prenet = PreNet(frame_dim*n_frames_per_step, [prenet_dim, prenet_dim])
        self.attention_rnn = nn.LSTMCell(prenet_dim+encoder_dim, attn_rnn_dim)
        self.attention = LocationAwareAttention(        
                            attn_rnn_dim,
                            attn_dim,
                            encoder_dim)
        self.decoder_rnn = nn.LSTMCell(attn_rnn_dim+encoder_dim, decoder_rnn_dim)
        self.linear_proj = Linear(decoder_rnn_dim+encoder_dim, frame_dim*n_frames_per_step)
        self.stop_proj = Linear(decoder_rnn_dim+encoder_dim, 1, bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, any_inputs):
        """Get the first empty frame"""
        B = any_inputs.size(0)
        any_inputs = torch.zeros(1, device=any_inputs.device).repeat(B, self.frame_dim*self.n_frames_per_step)
        return any_inputs

    def _reshape_decoder_inputs(self, decoder_inputs):
        """reshape according to the n_frames_per_step"""
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0), 
            decoder_inputs.size(1)//self.n_frames_per_step, -1
        )
        decoder_inputs = decoder_inputs.transpose(0, 1) # (B, len, n_mel) -> (len, B, n_mel)
        return decoder_inputs

    def _init_states(self, encoder_outputs, encoder_lengths):
        B = encoder_outputs.size(0)
        T = encoder_outputs.size(1)
        self.attn_rnn_hidden = torch.zeros(1, device=encoder_outputs.device).repeat(B, self.attn_rnn_dim)
        self.attn_rnn_cell = torch.zeros(1, device=encoder_outputs.device).repeat(B, self.attn_rnn_dim)
        self.decoder_rnn_hidden = torch.zeros(1, device=encoder_outputs.device).repeat(B, self.decoder_rnn_dim)
        self.decoder_rnn_cell = torch.zeros(1, device=encoder_outputs.device).repeat(B, self.decoder_rnn_dim)
        self.context = torch.zeros(1, device=encoder_outputs.device).repeat(B, self.encoder_dim)
        self.encoder_outputs = encoder_outputs
        self.encoder_lengths = encoder_lengths
        self.processed_encoder_outputs = self.attention.preprocess_source_inputs(encoder_outputs)
        self.attn_weights = torch.zeros(1, device=encoder_outputs.device).repeat(B, T)
        self.attn_weights_cum = torch.zeros(1, device=encoder_outputs.device).repeat(B, T)

    def _parse_decoder_outputs(self, mel_outputs, stop_tokens, alignments):
        """ stack outputs to be tensors
        """
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1).contiguous()
        alignments = torch.stack(alignments).transpose(0, 1).contiguous()
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.frame_dim)
        mel_outputs = mel_outputs.transpose(1, 2)
        return mel_outputs, stop_tokens, alignments


    def decode(self, decoder_input):
        attn_rnn_input = torch.cat((decoder_input, self.context), dim=-1)

        # run attn_rnn
        self.attn_rnn_hidden, self.attn_rnn_cell = self.attention_rnn(
            attn_rnn_input, (self.attn_rnn_hidden, self.attn_rnn_cell))

        self.attn_rnn_hidden = F.dropout(
            self.attn_rnn_hidden, self.p_attn_dropout, self.training
        )
        self.attn_rnn_cell = F.dropout(
            self.attn_rnn_cell, self.p_attn_dropout, self.training
        )
        # run the attention block
        attn_weights_cat = torch.cat(
            (self.attn_weights.unsqueeze(1),
            self.attn_weights_cum.unsqueeze(1)),
            dim = 1
        )
        self.context, self.attn_weights = self.attention(
            self.attn_rnn_hidden, self.encoder_outputs, self.processed_encoder_outputs,
            attn_weights_cat, self.encoder_lengths
        )
        self.attn_weights_cum += self.attn_weights

        # run the decoder rnn block
        decoder_rnn_input = torch.cat((self.attn_rnn_hidden, self.context), dim=-1)
        self.decoder_rnn_hidden, self.decoder_rnn_cell = self.decoder_rnn(
            decoder_rnn_input, (self.decoder_rnn_hidden, self.decoder_rnn_cell)
        )
        self.decoder_rnn_hidden = F.dropout(self.decoder_rnn_hidden, self.p_decoder_dropout, self.training)
        
        # get outputs
        decoder_hidden_context = torch.cat(
            (self.decoder_rnn_hidden, self.context), dim=1
        )
        mel_output = self.linear_proj(decoder_hidden_context)
        stop_token = self.stop_proj(decoder_hidden_context)
        return mel_output, self.attn_weights, stop_token

    def forward(
        self,
        encoder_outputs,
        decoder_inputs,
        encoder_lengths
    ):
        """Decoder forward for training
        Use teacher forcing

        Args: 
        encoder_outputs: encoder outputs
        decoder_inputs: Mel frames for teacher forcing
        encoder_lengths: encoder masks for padding

        Shapes:
            - encoder_outputs: (B, len, enc_out_dim)
            - decoder_inputs: (B, len_mel, mel_dim)
            - encoder_lengths: (B, len)
        """
        decoder_input = self.get_go_frame(encoder_outputs).unsqueeze(0)
        decoder_inputs = self._reshape_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0) # time first now
        decoder_inputs = self.prenet(decoder_inputs)

        self._init_states(encoder_outputs, encoder_lengths)

        mel_outputs, alignments, stop_tokens = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0)-1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, attention_weights, stop_token = self.decode(decoder_input)
            mel_outputs.append(mel_output.squeeze(1))
            alignments.append(attention_weights)
            stop_tokens.append(stop_token.squeeze(1))
        
        mel_outputs, stop_tokens, alignments = self._parse_decoder_outputs(
            mel_outputs, stop_tokens, alignments
        )
        return mel_outputs, stop_tokens, alignments
        
        
    def inference(
        self,
        encoder_outputs,
        encoder_lengths
    ):
        """ Decoder inference without teacher-forcing and use the stopnet to stop
        """
        decoder_input = self.get_go_frame(encoder_outputs)

        self._init_states(encoder_outputs, encoder_lengths)

        mel_outputs, alignments, stop_tokens = [], [], []
        i = 0
        while True:
            i += 1
            decoder_input = self.prenet(decoder_input)
            mel_output, attention_weights, stop_token = self.decode(decoder_input)
            stop_token = torch.sigmoid(stop_token.data)

            mel_outputs.append(mel_output.squeeze(1))
            stop_tokens.append(stop_token)
            alignments.append(attention_weights)

            if stop_token.squeeze(1) > self.stop_threshold:
                break
            if i == self.max_decoder_steps:
                print(f'Decoder stopped due to the max_decoder_steps -- {self.max_decoder_steps}')
                break
            # update decoder_input
            decoder_input = mel_output

        mel_outputs, stop_tokens, alignments = self._parse_decoder_outputs(
            mel_outputs, stop_tokens, alignments
        )
        return mel_outputs, stop_tokens, alignments
