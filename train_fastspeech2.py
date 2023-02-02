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
from src.models.fastspeech2 import FastSpeech2
from src.models.losses import FastSpeech2Loss
from src.trainer import Trainer
from src.utils.utils import sequence_mask

def compute_masks(lengths):
    """Compute masks against sequence paddings."""
    # B x T_in_max (boolean)
    masks = sequence_mask(lengths)
    return masks

preproc_cfg = PreprocessConfig()
model_cfg = ModelConfig()

metadata_train = load_samples_fs(
    preproc_cfg.preprocessed_path, 'train'
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

fastspeech = FastSpeech2(preproc_cfg, model_cfg)
loss = FastSpeech2Loss()

o = fastspeech(batch['token_ids'],  batch['token_ids_lengths'], batch['mel_lengths'], batch['pitch'], batch['energy'], batch['duration'])

print('model_outputs', o['model_outputs'].shape)
print('pitch_outputs', o['pitch_outputs'].shape)
print('energy_outputs', o['energy_outputs'].shape)
print('duration_outputs', o['duration_outputs'].shape)
print('mel_masks', o['mel_masks'].shape)

loss(
    mel_preds,
    pitch_preds,
    energy_preds,
    log_duration_preds,
    mel_targets,
    pitch_targets,
    energy_targets,
    duration_targets,
    src_masks,
    mel_masks,
)