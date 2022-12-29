import os
import numpy as np
from typing import Union, Tuple, List, Dict

#from src.dataset.dataset import TTSDataset

def parse_metadata(root_path, metadata_path: str):
    items = []
    with open(os.path.join(root_path, metadata_path), 'r', encoding='utf8') as f:
        for line in f:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[2].replace('\n', '')
            items.append({"text": text, "audio_file": wav_file, "root_path": root_path})
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

