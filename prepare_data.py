import os
# get the data
data_dir = "data"

from src.dataset import prepare_meldata_nvidia
from hparams import hparams as hps

#!wget -O $output_path/LJSpeech-1.1.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 
#tar -xf $data_dir/LJSpeech-1.1.tar.bz2 -C $data_dir

prepare_meldata_nvidia(data_dir, hps)