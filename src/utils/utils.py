import torch
import numpy as np

def _pad_data(x, length):
    _pad = 0
    assert x.ndim == 1
    return np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=_pad)

def prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

def _pad_tensor(x, length):
    _pad = 0.0
    assert x.ndim == 2
    x = np.pad(x, [[0, 0], [0, length - x.shape[1]]], mode="constant", constant_values=_pad)
    return x

def prepare_tensor(inputs, out_steps):
    max_len = max((x.shape[0] for x in inputs))
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_tensor(x.T, pad_len) for x in inputs])

def _pad_stop_target(x: np.ndarray, length: int, pad_val=1) -> np.ndarray:
    """Pad stop target array.
    Args:
        x (np.ndarray): Stop target array.
        length (int): Length after padding.
        pad_val (int, optional): Padding value. Defaults to 1.
    Returns:
        np.ndarray: Padded stop target array.
    """
    assert x.ndim == 1
    return np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=pad_val)


def prepare_stop_target(inputs, out_steps):
    """Pad row vectors with 1."""
    max_len = max((x.shape[0] for x in inputs))
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_stop_target(x, pad_len) for x in inputs])
    
# from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def sequence_mask(sequence_length, max_len=None):
    """Create a sequence mask for filtering padding in a sequence tensor.
    Args:
        sequence_length (torch.tensor): Sequence lengths.
        max_len (int, Optional): Maximum sequence length. Defaults to None.
    Shapes:
        - mask: :math:`[B, T_max]`
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    seq_range = torch.arange(max_len, dtype=sequence_length.dtype, device=sequence_length.device)
    # B x T_max
    mask = seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)
    return mask