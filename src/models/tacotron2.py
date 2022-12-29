from torch import nn

from src.models.layers import ConvBlock
from hparams import hparams as hps

class Tacotron2(nn.Module):
    def __init__(
        self,
        hps
    ):
        super(Tacotron2, self).__init__()
        self.encoder = Encoder(hps)

class Encoder(nn.Module):
    def __init__(
        self,
        hps
    ):
        super(Encoder, self).__init__()
        self.embeddings = nn.Embedding(hps.text_vocab_size, hps.embed_size)
        self.conv_blocks = nn.ModuleList([ConvBlock() for _ in range(hps.num_cnns)])
