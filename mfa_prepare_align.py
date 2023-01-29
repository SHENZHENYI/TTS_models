import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from src.utils.text import _clean_text


def prepare_align(config):
    in_dir = config.corpus_path
    out_dir = config.raw_path
    sampling_rate = config.sampling_rate
    max_wav_value = config.max_wav_value
    cleaners = config.text_cleaners
    speaker = "LJSpeech"
    with open(os.path.join(in_dir, "metadata.csv"), encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            base_name = parts[0]
            text = parts[2]
            text = _clean_text(text, cleaners)

            wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name))
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)

if __name__ == '__main__':
    from config.fastspeech2.preprocess import PreprocessConfig
    cfg = PreprocessConfig()
    prepare_align(cfg)