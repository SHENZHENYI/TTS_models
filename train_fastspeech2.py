import os
import numpy as np
import matplotlib.pyplot as plt
#!wget -O $output_path/LJSpeech-1.1.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 
#!tar -xf $output_path/LJSpeech-1.1.tar.bz2 -C $output_path
from torch.utils.data import DataLoader

from config.fastspeech2.preprocess import PreprocessConfig
from config.fastspeech2.model import ModelConfig

from src.dataset import load_samples_fs, TTSDataset
from src.utils.text import text_to_sequence
from src.models.fastspeech2_modules import Encoder, VariancePredictor, LengthRegulator, VarianceAdaptor
from src.trainer import Trainer
from src.utils.utils import sequence_mask

def compute_masks(lengths):
    """Compute masks against sequence paddings."""
    # B x T_in_max (boolean)
    masks = sequence_mask(lengths)
    return masks
#prepare_meldata(data_dir)
"""
mel = np.load(os.path.join(data_dir, 'mels', 'LJ001-0001.npy'))
print(mel.shape)
plt.imshow(mel, cmap='hot', interpolation='nearest')
plt.savefig('./tmp.png')
"""

cfg = PreprocessConfig()
model_cfg = ModelConfig()

metadata_train = load_samples_fs(
    cfg.preprocessed_path, 'train'
)

train_ds = TTSDataset(metadata_train)

train_dl = DataLoader(
            train_ds,
            batch_size=16,
            shuffle=False,
            collate_fn=train_ds.collate_fn,
            drop_last=False, 
            num_workers=0,
            pin_memory=True,
        )

for batch in train_dl:
    break 

import torch
device = torch.device('cpu')

encoder = Encoder(
    n_src_vocab=cfg.phoneme_vocab_size,
    max_seq_len=model_cfg.max_seq_len,
    n_layers=model_cfg.encoder_layer,
    encoder_hidden_dim=model_cfg.encoder_hidden,
    n_head=model_cfg.encoder_head,
    conv_kernel_size=model_cfg.encoder_conv_kernel_size,
    conv_filter_size=model_cfg.encoder_conv_filter_size,
).to(device)

vp = VariancePredictor(
    input_dim=model_cfg.encoder_hidden,
    conv_filter_size=model_cfg.variance_predictor_conv_filter_size,
    conv_kernel_size=model_cfg.variance_predictor_conv_kernel_size,
    dropout=model_cfg.variance_predictor_dropout
).to(device)

va = VarianceAdaptor(
    input_dim=model_cfg.encoder_hidden,
    conv_filter_size=model_cfg.variance_predictor_conv_filter_size,
    conv_kernel_size=model_cfg.variance_predictor_conv_kernel_size,
    dropout=model_cfg.variance_predictor_dropout,
    pitch_scale=model_cfg.variance_predictor_pitch_scale,
    energy_scale=model_cfg.variance_predictor_energy_scale,
    n_bins=model_cfg.n_bins,
    pitch_energy_stats_path=os.path.join(cfg.preprocessed_path, 'stats.json'),
).to(device)


tokens_mask = compute_masks(batch['token_ids_lengths']).to(device)

o = encoder(batch['token_ids'].to(device), tokens_mask)
#o = vp(o)
#print(o.shape)
duration = batch['duration'].to(device)
max_mel_len = max(batch['mel_lengths']).item()
pitch = batch['pitch']
energy = batch['energy']

print(o.shape, duration.shape, max_mel_len)


o, log_duration_prediction, pitch_prediction, energy_prediction, mel_len = va(o, tokens_mask, max_mel_len, pitch, energy, duration)
print(o.shape)