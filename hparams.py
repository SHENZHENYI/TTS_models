class hparams:
    # audio
    num_mels = 80
    num_freq = 513
    sample_rate = 22050
    frame_shift = 256
    frame_length = 1024
    fmin = 125
    fmax = 7600
    power = 1.5
    gl_iters = 30

    # tacotron2
    # encoder
    text_vocab_size = 148
    embed_size = 512
    cnn_channels = 512
    cnn_kernel_size = 5
    num_cnns = 3
