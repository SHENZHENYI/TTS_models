import os
import numpy as np
import matplotlib.pyplot as plt
# get the data
data_dir = "data"
checkpoint_path = "save_dir/tacotron_fold0_e358.pth"
#!wget -O $output_path/LJSpeech-1.1.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 
#!tar -xf $output_path/LJSpeech-1.1.tar.bz2 -C $output_path

from hparams import hparams as hps
from src.dataset import load_samples, prepare_meldata
from src.utils.text import text_to_sequence
from src.models.tacotron2 import Tacotron2
from src.trainer import Trainer

#prepare_meldata(data_dir)
"""
mel = np.load(os.path.join(data_dir, 'mels', 'LJ001-0001.npy'))
print(mel.shape)
plt.imshow(mel, cmap='hot', interpolation='nearest')
plt.savefig('./tmp.png')
"""

metadata_train, metadata_val = load_samples(
    data_dir, 'metadata.csv', True, 50
)


model = Tacotron2(hps)

trainer = Trainer(
    hps,
    model,
    fold=0,
    train_samples=metadata_train,
    val_samples=metadata_val,
    test_samples=None,
    device='cpu',
    checkpoint_path=None
)


trainer.fit()


"""
inferencer = Trainer(
    hps,
    model,
    fold=0,
    train_samples=None, #metadata_val[:5],
    val_samples=None,#metadata_val[:5],
    test_samples=metadata_val[:2],
    device='cpu',
    checkpoint_path=checkpoint_path
)

preds = inferencer.inference()
#print(preds['stop_tokens'].all[0])
#print(preds['alignments'].all[0].shape)
plt.imshow(preds['alignments'].all[0].T.numpy(), cmap='hot', interpolation='nearest')
plt.savefig('./tmp_pred.png')
"""

