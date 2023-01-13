import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Dict, Tuple, List

from hparams import hparams as hps
from src.models.basemodel import BaseModel
from src.models.tacotron2_modules import *
from src.models.losses import TacotronLoss
from src.dataset.dataset import TTSDataset
from src.utils.utils import sequence_mask
from src.trainer_utils import get_scheduler

class Tacotron2(BaseModel):
    def __init__(
        self,
        hps
    ):
        super(Tacotron2, self).__init__()
        self.cfg = hps
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
        self.postnet = PostNet(hps.num_mels*hps.n_frames_per_step, hps.postnet_n_convs)
    
    def compute_masks(self, lengths):
        """Compute masks against sequence paddings."""
        # B x T_in_max (boolean)
        masks = sequence_mask(lengths)
        return masks

    @staticmethod
    def _reshape_outputs(decoder_outputs, postnet_outputs):
        return decoder_outputs.transpose(1, 2), postnet_outputs.transpose(1, 2)

    def forward(
        self, text, text_lengths, mel, mel_lengths=None
    ):
        """
        Shapes:
            - model_outputs: (B, mel_len, mel_dim)
            - decoder_outputs: (B, mel_len, mel_dim)
            - alignments: (B, mel_len, encoder_len)
            - stop_tokens: (B, mel_len)
        """
        # compute masks
        text_masks = self.compute_masks(text_lengths)
        mel_masks = self.compute_masks(mel_lengths)

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
        if mel_masks is not None:
            decoder_outputs = decoder_outputs * mel_masks.unsqueeze(1).expand_as(decoder_outputs)

        # postnet learns the residual
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = postnet_outputs + decoder_outputs
        if mel_masks is not None:
            postnet_outputs = postnet_outputs * mel_masks.unsqueeze(1).expand_as(postnet_outputs)
        
        # process the output shapes
        decoder_outputs, postnet_outputs = self._reshape_outputs(decoder_outputs, postnet_outputs)
        
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
        mel_inputs = batch['mel'] # (B, T, 80)
        mel_lengths = batch['mel_lengths']
        mel_masks = self.compute_masks(mel_lengths)
        stop_targets = batch['stop_targets']
        outputs = self.forward(text_input, text_lengths, mel_inputs, mel_lengths)

        loss_dict = criterion(
            outputs['model_outputs'],
            outputs['decoder_outputs'],
            mel_masks,
            mel_lengths,
            outputs['stop_tokens'],
            mel_inputs,
            stop_targets
        )
        return outputs, loss_dict
    
    @torch.no_grad()
    def inference(self, batch):
        """ Inference step
        forward process without teacher forcing
        """
        # get inputs
        text = batch['token_ids']
        text_lengths = batch['token_ids_lengths']
        text_masks = self.compute_masks(text_lengths)

        # encoder
        emb = self.embeddings(text).transpose(1, 2)
        encoder_outputs = self.encoder(emb, text_lengths)
        encoder_outputs = encoder_outputs * text_masks.unsqueeze(2).expand_as(encoder_outputs)

        # decoder
        decoder_outputs, stop_tokens, alignments = self.decoder.inference(encoder_outputs, text_masks)
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs

        decoder_outputs, postnet_outputs = self._reshape_outputs(decoder_outputs, postnet_outputs)
        outputs = {
            "model_outputs": postnet_outputs,
            "decoder_outputs": decoder_outputs,
            "alignments": alignments,
            "stop_tokens": stop_tokens,
        }
        return outputs

    @torch.no_grad()
    def eval_step(self, batch: Dict, criterion) -> Tuple[Dict, Dict]:
        return self.train_step(batch, criterion)

    def get_train_loader(self, samples: Dict, tokenizer=None):
        train_dataset = TTSDataset(samples)
        train_dataset.preprocess_data()
        return DataLoader(
            train_dataset,
            batch_size=hps.batch_size,
            shuffle=self.cfg.shuffle,
            collate_fn=train_dataset.collate_fn,
            drop_last=False, 
            num_workers=hps.num_loader_workers,
            pin_memory=True,
        )
    
    def get_val_loader(self, samples: Dict, tokenizer=None):
        val_dataset = TTSDataset(samples)
        return DataLoader(
            val_dataset,
            batch_size=hps.batch_size*2,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            drop_last=False, 
            num_workers=hps.num_loader_workers,
            pin_memory=True,
        )

    def get_test_loader(self, samples: Dict, tokenizer=None):
        test_dataset = TTSDataset(samples)
        return DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=test_dataset.collate_fn,
            drop_last=False, 
            num_workers=hps.num_loader_workers,
            pin_memory=True,
        )

    def get_criterion(self) -> nn.Module:
        return TacotronLoss()
    
    def get_optimizer(self):
        return AdamW(self.parameters(), lr=hps.learning_rate, eps=hps.eps, \
                        betas=hps.betas, weight_decay=hps.weight_decay)

    def get_tokenizer(self, instantiate=False):
        return None

    def get_scheduler(self, optimizer: object, num_train_steps: int):
        return  get_scheduler(self.cfg.scheduler, optimizer, num_train_steps, self.cfg.warmup_ratio, self.cfg.num_cycles)

    def get_metric(self,):
        return TacotronLoss()