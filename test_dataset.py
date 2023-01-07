import os
import numpy as np
import matplotlib.pyplot as plt
# get the data
data_dir = "data"

from hparams import hparams as hps
from src.dataset import load_samples, prepare_meldata, TTSDataset
from src.utils.text import text_to_sequence
from src.models.tacotron2 import Tacotron2
from src.trainer import Trainer

metadata_train, metadata_val = load_samples(
    data_dir, 'metadata.csv', True, 50
)

ds = TTSDataset(metadata_train)

print(len(ds))

ds.preprocess_data()
