from dataclasses import asdict, dataclass, field

@dataclass
class PreprocessConfig:
    # validation size
    val_size: int = 50

    # text
    phoneme_vocab_size = 172
    corpus_path: str = './data'
    lexicon_path: str = './mfa_data/librispeech-lexicon.txt'
    raw_path: str = "./data/raw_data/LJSpeech"
    preprocessed_path: str = "./data/preprocessed_data/LJSpeech"
    text_cleaners = ['english_cleaners']

    # audio
    sampling_rate: int = 22050
    max_wav_value: float = 32768.0

    # stft
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024

    # mel
    n_mel_channels: int = 80
    mel_fmin: int = 0
    mel_fmax: int = 8000

    # pitch
    pitch_feature: str = 'frame_level'
    pitch_norm: bool = False 

    # energy
    energy_feature: str = 'frame_level'
    energy_norm: bool = False