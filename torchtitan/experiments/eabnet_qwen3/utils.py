# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import scipy.signal as signal
from einops import rearrange
import numpy as np
from functools import partial
import json

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
    # y = torch.stft(x, 256, win_shift, win_size, torch.hann_window(256).to(x.device),return_complex=True)
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


def resample(audio, current_sr, target_sr):
    n = audio.shape[0]
    if len(audio.shape)==2:
        assert audio.shape[1]<10, ('maybe (not) an error, check audio shape', audio.shape)
    if current_sr!=target_sr:
        audio = signal.resample(audio, int(n*target_sr/current_sr))
    return audio


MODEL_TO_DEVICE = 'model to device'
DEVICE_TO_MODEL = 'device to model'

def cal_take_indices(src_mics, tgt_mics):
    assert len(src_mics)==len(tgt_mics), (src_mics,tgt_mics)
    take_indices = []
    for p in tgt_mics:
        found = False
        for i,s in enumerate(src_mics):
            if p == s:
                take_indices.append(i)
                found = True
                break
        assert found, (src_mics, tgt_mics, p)
    assert len(set(take_indices))==len(tgt_mics), (src_mics,tgt_mics,take_indices)
    return take_indices


def converter(wav, take_indices):
    assert len(wav.shape) == 2, wav.shape
    assert wav.shape[1] == len(take_indices), (wav.shape,'expect wav to be (n,c)')
    return np.take(wav, take_indices, axis=1)


def make_converter(type, mcse_settings_path, device_mics):
    with open(mcse_settings_path, 'r') as f:
        opt = json.load(f)
    model_mics = opt['mic_array']['mics']
    if type == MODEL_TO_DEVICE:
        take_indices = cal_take_indices(model_mics, device_mics)
    elif type == DEVICE_TO_MODEL:
        take_indices = cal_take_indices(device_mics, model_mics)
    else:
        raise ValueError(type)
    
    print(f'make converter, type: {type}, take_indices: {take_indices}')
    
    return partial(converter, take_indices=take_indices)


if __name__ == "__main__":
    stft_params = {'sr': 16000,
                'win_size_sec': 0.02,
                'win_shift_sec': 0.01,
                'num_fft': 320}

    aa = torch.rand(1,8, 16000*12)
    print(stft_then_sqrt_comp(aa, True, **stft_params).shape)