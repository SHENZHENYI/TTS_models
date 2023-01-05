from dataclasses import asdict, dataclass, field

@dataclass
class hparams:
    # file paths
    model: str = 'tacotron'
    save_dir: str = 'save_dir'

    # audio
    num_mels: int = 80
    n_frames_per_step: int = 1
    num_freq: int = 513
    sample_rate: int = 22050
    frame_shift: int = 256
    frame_length: int = 1024
    fmin: int = 125
    fmax: int = 7600
    power: float = 1.5
    gl_iters: int = 30

    # model
    # encoder
    text_vocab_size: int = 148
    embed_size: int = 512
    cnn_channels: int = 512
    cnn_kernel_size: int = 5
    num_cnns: int = 3
    lstm_hidden_size: int = 512
    # decoder
    attn_rnn_dim: int = 1024
    decoder_rnn_dim = 1024
    prenet_dim = 256
    attn_dim = 128
    p_attn_dropout = 0.1
    p_decoder_dropout = 0.1
    # postnet 
    postnet_n_convs = 5

    # training
    device: str = 'cuda'
    epoch: int = 100
    batch_size: int = 64
    num_loader_workers: int = 8
    learning_rate: float = 1e-4
    betas = [0.9, 0.999]
    eps: float = 1e-6
    weight_decay: float = 1e-6
    apex: bool = True
    max_grad_norm: float = 25.
    print_freq: int = 10
