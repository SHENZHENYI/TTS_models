import os
import random
from typing import Dict, List, Union

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from src.utils.text import text_to_sequence
from src.utils.audio import melspectrogram
from src.utils.utils import prepare_data, prepare_tensor, prepare_stop_target

class TTSDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
    ):
        self.samples = samples
        self.sample_rate = 22050
        self.n_frames_per_step = 1
    
    def __len__(self,):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.load_data(idx)

    def load_wav(self, filename):
        wav, sr = librosa.load(filename, sr=self.sample_rate)
        return wav 
    
    def load_melspectrogram(self, filename):
        return np.load(filename)
    
    def melspectrogram(self, wav):
        return melspectrogram(wav)
    
    def get_token_ids(self, text):
        return text_to_sequence(text, ['english_cleaners'])

    def load_data(self, idx):
        item = self.samples[idx]

        raw_text = item['text']

        wav = np.asarray(self.load_wav(item['audio_file']))

        mel = self.load_melspectrogram(item['mel_file'])

        token_ids = np.array(self.get_token_ids(raw_text))

        return {
            "raw_text": raw_text,
            "token_ids": token_ids,
            "wav": wav,
            "mel": mel
        }

    @staticmethod
    def _sort_batch(batch, text_lengths):
        text_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor(text_lengths), dim=0, descending=True)
        batch = [batch[idx] for idx in ids_sorted_decreasing]
        return batch, text_lengths, ids_sorted_decreasing
    
    def collate_fn(self, batch):
        token_ids_lengths = np.array([len(d["token_ids"]) for d in batch])
        batch, token_ids_lengths, _ = self._sort_batch(batch, token_ids_lengths)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}
        token_ids = prepare_data(batch['token_ids'])
        mel_lengths = [m.shape[1] for m in batch['mel']]
        stop_targets = [np.array([0.0] * (mel_len - 1) + [1.0]) for mel_len in mel_lengths]
        stop_targets = prepare_stop_target(stop_targets, out_steps=self.n_frames_per_step)
        mel = prepare_tensor(batch['mel'], 1)
        mel = mel.transpose(0, 2, 1)

        wav = None #prepare_data(batch['wav'])

        #token_ids_lengths = torch.LongTensor(token_ids_lengths)
        token_ids = torch.LongTensor(token_ids)
        token_ids_lengths = torch.LongTensor(token_ids_lengths)
        mel = torch.FloatTensor(mel).contiguous()
        mel_lengths = torch.LongTensor(mel_lengths)
        #wav = torch.FloatTensor(wav)
        stop_targets = torch.FloatTensor(stop_targets)

        return {
            'token_ids': token_ids,
            'token_ids_lengths': token_ids_lengths,
            'mel': mel,
            'mel_lengths': mel_lengths,
            'stop_targets': stop_targets,
            'wav': wav,
            'raw_text': batch['raw_text']
        }