"""
largely adopted from https://github.com/coqui-ai/TTS/blob/f814d523945fc43071d037a1fb9edcdad99949b2/TTS/tts/datasets/dataset.py
"""

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
        self.max_audio_len = 10 * 22050
        self.min_audio_len = 1 * 22050
        self.max_text_len = float('inf')
        self.min_text_len = 1
    
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
    
    @staticmethod
    def _compute_lengths(samples):
        """ Compute audio and text lengths
        """
        new_samples = []
        for item in samples:
            audio_length = os.path.getsize(item["audio_file"]) / 16 * 8  # assuming 16bit audio
            text_lenght = len(item["text"])
            item["audio_length"] = int(audio_length)
            item["text_length"] = int(text_lenght)
            new_samples += [item]
        return new_samples

    @staticmethod
    def _filter_by_length(lengths: List[int], min_len: int, max_len: int):
        """ filter out lengths out of the range of (min_len, max_len)
        returns ignore_idx and keep_idxs
        """
        idxs = np.argsort(lengths)  # ascending order
        ignore_idx = []
        keep_idx = []
        for idx in idxs:
            length = lengths[idx]
            if length < min_len or length > max_len:
                ignore_idx.append(idx)
            else:
                keep_idx.append(idx)
        return ignore_idx, keep_idx

    @staticmethod
    def _select_samples_by_idx(idxs, samples):
        samples_new = []
        for idx in idxs:
            samples_new.append(samples[idx])
        return samples_new

    @staticmethod
    def _sort_by_audio_length(samples: List[List]):
        audio_lengths = [s["audio_length"] for s in samples]
        idxs = np.argsort(audio_lengths)  # ascending order
        return idxs

    def preprocess_data(self, verbose=True):
        """ 
        filter the samples by min and max text & audio lens, 
        and sort the samples by audio lens
        """
        samples = self._compute_lengths(self.samples)

        text_lengths = [s["text_length"] for s in samples]
        audio_lengths = [s["audio_length"] for s in samples]
        text_ignore_idx, text_keep_idx = self._filter_by_length(text_lengths, self.min_text_len, self.max_text_len)
        audio_ignore_idx, audio_keep_idx = self._filter_by_length(audio_lengths, self.min_audio_len, self.max_audio_len)

        keep_idx = list(set(text_keep_idx) & set(audio_keep_idx))
        ignore_idx = list(set(text_ignore_idx) & set(audio_ignore_idx))

        # filter out samples out of the specified lens
        samples = self._select_samples_by_idx(keep_idx, samples)

        sorted_idxs = self._sort_by_audio_length(samples)

        samples = self._select_samples_by_idx(sorted_idxs, samples)

        text_lengths = [s["text_length"] for s in samples]
        audio_lengths = [s["audio_length"] for s in samples]
        self.samples = samples

        if verbose:
            print("Preprocessing data...")
            print(f"Max text len: {np.max(text_lengths)}")
            print(f"Min text len: {np.min(text_lengths)}")
            print(f"Mean text len: {np.mean(text_lengths)}")
            print(f"Max audio len: {np.max(audio_lengths)}")
            print(f"Min audio len: {np.min(audio_lengths)}")
            print(f"Mean audio len: {np.mean(audio_lengths)}")
            print(f"Discarded number of samples: {len(ignore_idx)}")
            print(f"Total number of samples: {len(audio_lengths)}")

