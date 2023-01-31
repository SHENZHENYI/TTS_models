from dataclasses import asdict, dataclass, field

@dataclass
class ModelConfig:
    # encoder
    encoder_layer = 4
    encoder_head = 2
    encoder_hidden = 256
    encoder_conv_filter_size = 1024
    encoder_conv_kernel_size = [9, 1]

    variance_predictor_conv_filter_size = 256
    variance_predictor_conv_kernel_size = 3
    variance_predictor_dropout = 0.5
    variance_predictor_pitch_scale = 'log'
    variance_predictor_energy_scale = 'linear'
    variance_predictor_n_bins = 256

    max_seq_len = 1000
