import os
import random
from typing import Dict, List, Union

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from src.utils.text import text_to_sequence
from src.utils.audio import melspectrogram

class TTSDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
    ):
        self.samples = samples
        self.sample_rate = 22050
    
    def __len__(self,):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.load_data(idx)

    def load_wav(self, filename):
        wav, sr = librosa.load(filename, sr=self.sample_rate)
        return wav 
    
    def melspectrogram(self, wav):
        return melspectrogram(wav)
    
    def get_token_ids(self, text):
        return text_to_sequence(text, ['english_cleaners'])

    def load_data(self, idx):
        item = self.samples[idx]

        raw_text = item['text']

        wav = np.asarray(self.load_wav(item['audio_file']))

        mel = self.melspectrogram(wav)

        token_ids = self.get_token_ids(raw_text)

        return {
            "raw_text": raw_text,
            "token_ids": token_ids,
            "wav": wav,
            "mel": mel
        }
    
    def collate_fn(self, batch):
        pass