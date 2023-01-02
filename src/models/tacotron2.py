import torch
from torch import nn
from torch.nn import functional as F

from hparams import hparams as hps
from src.models.layers import ConvBlock, Linear, LocationAwareAttention
from src.utils.utils import sequence_mask

class Tacotron2(nn.Module):
    def __init__(
        self,
        hps
    ):
        super(Tacotron2, self).__init__()
        self.embeddings = nn.Embedding(hps.text_vocab_size, hps.embed_size, padding_idx=0)
        self.encoder = Encoder(
            hps.cnn_channels,
            hps.cnn_kernel_size,
            hps.num_cnns,
            hps.lstm_hidden_size
        )
        self.decoder = Decoder(
            hps.lstm_hidden_size,
            hps.n_frames_per_step,
            hps.num_mels,
            hps.attn_rnn_dim,
            hps.decoder_rnn_dim,
            hps.prenet_dim,
            hps.attn_dim,
            hps.p_attn_dropout,
            hps.p_decoder_dropout,
        )
        self.postnet = PostNet()
    
    def compute_masks(self, text_lengths, mel_lengths):
        """Compute masks against sequence paddings."""
        # B x T_in_max (boolean)
        input_mask = sequence_mask(text_lengths)
        output_mask = sequence_mask(mel_lengths)
        return input_mask, output_mask

    def forward(
        self, text, text_lengths, mel, mel_lengths
    ):
        # compute masks
        text_masks, mel_masks = self.compute_masks(text_lengths, mel_lengths)

        # encoder side
        emb = self.embeddings(text).transpose(1, 2)
        encoder_outputs = self.encoder(emb, text_lengths)
        encoder_outputs = encoder_outputs * text_masks.unsqueeze(2).expand_as(encoder_outputs)

        # decoder side
        decoder_outputs, stop_tokens, alignments = self.decoder(
            encoder_outputs,
            mel,
            text_masks,
        )
        if mel_lengths is not None:
            decoder_outputs = decoder_outputs * mel_masks.unsqueeze(1).expand_as(decoder_outputs)

        # postnet learns the residual
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = postnet_outputs + decoder_outputs

        return {
            'model_outputs': postnet_outputs,
            'decoder_outputs': decoder_outputs,
            'alignments': alignments,
            'stop_tokens': stop_tokens,
        }
    
    def train_step(self, batch, criterion):
        """ training step
        """
        text_input = batch['token_ids']
        text_lengths = batch['token_ids_lengths']
        mel_inputs = batch['mel']
        mel_lengths = batch['mel_lengths']
        stop_targets = batch['stop_targets']

        outputs = self.forward(text_input, text_lengths, mel_inputs, mel_lengths)
    
    @torch.no_grad()
    def inference(self, batch):
        """ Inference step
        """

class Encoder(nn.Module):
    def __init__(
        self,
        cnn_channels,
        cnn_kernel_size,
        num_cnns,
        lstm_hidden_size,
    ):
        super(Encoder, self).__init__()
        self.conv_blocks = nn.ModuleList([ConvBlock(cnn_channels, cnn_kernel_size) for _ in range(num_cnns)])
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden_size//2, \
                        num_layers=1, batch_first=True, bias=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.conv_blocks:
            x = conv(x)
        x = x.transpose(1, 2)
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x

class PreNet(nn.Module):
    def __init__(
        self,
        in_size: int,
        sizes: list[int]
    ):
        super(PreNet, self).__init__()
        in_sizes = [in_size] + sizes[:-1]
        self.layers = nn.ModuleList([
            Linear(in_size, out_size) for in_size, out_size in zip(in_sizes, sizes)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x).relu()
        return x

class PostNet(nn.Module):
    def __init__(
        self,
        channels,
        n_convs
    ):
        super(PostNet, self).__init__()
        self.convs = nn.ModuleList([
            ConvBlock(channels, 512, kernel_size=5, activation='tanh') for _ in n_convs-1
        ])
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
    ):
        super(Decoder, self).__init__()
        self.n_frames_per_step = n_frames_per_step
        self.frame_dim = frame_dim
        self.encoder_dim = encoder_dim
        self.p_attn_dropout = p_attn_dropout
        self.p_decoder_dropout = p_decoder_dropout

        self.prenet = PreNet(frame_dim*n_frames_per_step, [prenet_dim, prenet_dim])
        self.postnet = PostNet()
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
            decoder_inputs.size(1)/self.n_frames_per_step, -1
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
        decoder_inputs = torch.cat(decoder_input, decoder_inputs, dim=0) # time first now
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
        encoder_outpus,
        decoder_inputs,
        encoder_lengths
    ):
        pass

    