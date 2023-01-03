import os
# get the data
output_path = "data"

#!wget -O $output_path/LJSpeech-1.1.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 
#!tar -xf $output_path/LJSpeech-1.1.tar.bz2 -C $output_path

from hparams import hparams as hps
from src.dataset import load_samples
from src.utils.text import text_to_sequence
from src.models.tacotron2 import Tacotron2
from src.trainer import Trainer


metadata_train, metadata_val = load_samples(
    'data', 'metadata.csv', True, 50
)

model = Tacotron2(hps)

trainer = Trainer(
    hps,
    model,
    fold=0,
    train_samples=metadata_val,
    val_samples=metadata_val,
    test_samples=None,
    device='cpu',
    checkpoint_path=None
)

trainer.fit()