from dataclasses import asdict, dataclass, field

@dataclass
class ModelConfig:
    # encoder
    encoder_layer = 4
    encoder_head = 2
    encoder_hidden = 256
    conv_filter_size = 1024
    conv_kernel_size = [9, 1]

    max_seq_len = 1000
