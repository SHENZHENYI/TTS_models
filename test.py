import os
# get the data
output_path = "data"

#!wget -O $output_path/LJSpeech-1.1.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 
#!tar -xf $output_path/LJSpeech-1.1.tar.bz2 -C $output_path

from hparams import hparams as hps
from src.dataset import load_samples
from src.dataset.dataset import TTSDataset
from src.utils.text import text_to_sequence
from src.models.tacotron2 import Tacotron2

from torch.utils.data import DataLoader


metadata_train, metadata_eval = load_samples(
    'data', 'metadata.csv', True, 50
)

eval_ds = TTSDataset(metadata_eval, )

eval_loader = DataLoader(
    eval_ds,
    batch_size=4,#hps.batch_size,
    shuffle=False,  # if there is no other sampler
    collate_fn=eval_ds.collate_fn,
    drop_last=False,  # setting this False might cause issues in AMP training.
    num_workers=0,
    pin_memory=False,
)

for batch in eval_loader:
    break

print(batch)

model = Tacotron2(hps)

#out = model(batch['token_ids'], batch['token_ids_lengths'])

#print(out.shape)