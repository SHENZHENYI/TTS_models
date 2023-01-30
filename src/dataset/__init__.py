import os
import numpy as np
from typing import Union, Tuple, List, Dict

#from src.utils.audio import melspectrogram, load_wav
from src.dataset.dataset_v2 import *
from src.utils.audio.stft import TacotronSTFT
from src.dataset.utils import load_mel_from_wav

def prepare_meldata(root_path, hps):
    """Create a folder of mel80 decoded from wavs"""
    wav_dir = os.path.join(root_path, "wavs")
    os.makedirs(os.path.join(root_path, "mels"), exist_ok=True)
    for f in os.listdir(wav_dir):
        mel_path = os.path.join(root_path, "mels", f.split('.')[0]+'.npy')
        wav = np.asarray(load_wav(os.path.join(wav_dir, f)))
        mel = melspectrogram(wav)
        np.save(mel_path, mel)

def prepare_meldata_nvidia(root_path, hps):
    """Create a folder of mel80 decoded from wavs"""
    stft = TacotronSTFT(
            hps.filter_length, hps.hop_length, hps.win_length,
            hps.num_mels, hps.sample_rate, hps.mel_fmin,
            hps.mel_fmax)
    wav_dir = os.path.join(root_path, "wavs")
    os.makedirs(os.path.join(root_path, "mels"), exist_ok=True)
    for f in os.listdir(wav_dir):
        mel_path = os.path.join(root_path, "mels", f.split('.')[0]+'.npy')
        mel = load_mel_from_wav(os.path.join(wav_dir, f), stft, hps.max_wav_value).numpy()
        np.save(mel_path, mel)
        print(f'Saved {f}')

def prepare_duration_pitch_energy_mel_metadata_fastspeech2():
    from src.utils.preprocessor.fastspeech_preproc import Preprocessor
    from config.fastspeech2.preprocess import PreprocessConfig

    cfg = PreprocessConfig()
    preproc = Preprocessor(cfg)

    preproc.build_from_path()


def parse_metadata(root_path, metadata_path: str):
    items = []
    with open(os.path.join(root_path, metadata_path), 'r', encoding='utf8') as f:
        for line in f:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            mel_file = os.path.join(root_path, "mels", cols[0] + ".npy")
            text = cols[2].replace('\n', '')
            items.append({"text": text, "audio_file": wav_file, "root_path": root_path, 'mel_file': mel_file})
    return items

def parse_metadata_fs(root_path, data_split: str = 'train'):
    items = []
    with open(os.path.join(root_path, f'{data_split}.txt'), 'r', encoding='utf8') as f:
        for line in f:
            filename, speaker, phoneme, text = line.strip("\n").split("|")
            wav_path = os.path.join(os.path.dirname(os.path.dirname(root_path)), "wavs", filename + ".wav")
            mel_path = os.path.join(root_path, "mel", f"{speaker}-mel-{filename}.npy")
            duration_path = os.path.join(root_path, "duration", f"{speaker}-duration-{filename}.npy")
            pitch_path = os.path.join(root_path, "pitch", f"{speaker}-pitch-{filename}.npy")
            energy_path = os.path.join(root_path, "energy", f"{speaker}-energy-{filename}.npy")
            items.append({
                "text": text,
                "phoneme": phoneme,
                "wav_path": wav_path,
                "mel_path": mel_path,
                "duration_path": duration_path,
                "pitch_path": pitch_path,
                "energy_path": energy_path
            })
    return items


def split_data(all_data, eval_split_size):
    np.random.seed(0)
    np.random.shuffle(all_data)
    return all_data[:eval_split_size], all_data[eval_split_size:]

def load_samples(
    root_dir: str,
    metadata_path: str,
    eval_split: bool,
    eval_split_size: int
) -> Tuple[List[Dict], List[Dict]]:
    metadata_train = parse_metadata(root_dir, metadata_path)
    metadata_eval = []
    if eval_split:
        metadata_eval, metadata_train = split_data(metadata_train, eval_split_size)
    return metadata_train, metadata_eval

def load_samples_fs(
    root_dir: str,
    data_split: str = 'train'
) -> List[Dict]:
    return parse_metadata_fs(root_dir, data_split)