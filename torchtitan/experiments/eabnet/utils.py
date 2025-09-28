# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch
from einops import rearrange

def sqrt_compress(x):
    '''
    x: b, f, t, 2
    '''
    # conduct sqrt power-compression
    mag, phase = torch.norm(x, dim=-1) ** 0.5, torch.atan2(x[..., -1], x[..., 0])
    return torch.stack((mag * torch.cos(phase), mag * torch.sin(phase)), dim=-1)

def sqrt_decompress(x):
    '''
    x: b, f, t, 2
    '''
    mag, phase = torch.norm(x, dim=-1) ** 2, torch.atan2(x[..., -1], x[..., 0])
    return torch.stack((mag * torch.cos(phase), mag * torch.sin(phase)), dim=-1)

def stft_then_sqrt_comp(x, is_multi_channel, sr, win_size_sec, win_shift_sec, num_fft):
    '''
    Params:
        x: b, [m,] s
    Returns:
        y: b, [m,] f, t, 2
    '''

    if is_multi_channel:
        b, m, s = x.shape
        x = x.reshape(b*m, s)
    else:
        b, s = x.shape
    
    win_size = int(win_size_sec * sr)
    win_shift = int(win_shift_sec * sr)

    y = torch.stft(x, num_fft, win_shift, win_size, torch.hann_window(win_size).to(x.device),return_complex=True)
    y = torch.view_as_real(y)
    y = sqrt_compress(y)

    if is_multi_channel:
        y = rearrange(y, '(b m) f t c -> b m f t c', m=m)

    return y

def sqrt_decomp_then_istft(x, is_multi_channel, sr, win_size_sec, win_shift_sec, num_fft):
    '''
    Params:
        x: b, [m,] f, t, 2
    Returns:
        y: b, [m,] s
    '''
    if is_multi_channel:
        b, m, f, t, _ = x.shape
        x = x.reshape(b*m, f, t, 2)
    else:
        b, f, t, _ = x.shape

    y = sqrt_decompress(x)
    
    win_size = int(win_size_sec * sr)
    win_shift = int(win_shift_sec * sr)
    
    y = torch.istft(torch.view_as_complex(y.contiguous()), num_fft, win_shift, win_size, torch.hann_window(win_size).to(x.device))

    if is_multi_channel:
        y = rearrange(y, '(b m) s -> b m s', m=m)
    
    return y

