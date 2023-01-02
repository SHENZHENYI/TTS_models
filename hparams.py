class hparams:
    # audio
    num_mels = 80
    n_frames_per_step = 1
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
    lstm_hidden_size = 512

    # decoder
    attn_rnn_dim = 1024
    decoder_rnn_dim = 1024
    prenet_dim = 256
    attn_dim = 128
    p_attn_dropout = 0.1
    p_decoder_dropout = 0.1

    # training
    batch_size = 4
    num_loader_workers = 1
