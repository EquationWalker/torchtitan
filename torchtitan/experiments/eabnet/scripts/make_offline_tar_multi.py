import os
import json
import soundfile as sf
from tqdm import tqdm
import numpy as np
import pyroomacoustics as pra
from einops import rearrange
import torch
from torch import Tensor
from scipy.signal import fftconvolve
import scipy.signal as signal
import webdataset as wds
from pathlib import Path
import multiprocessing as mp

EPS = np.finfo(float).eps


def sqrt_compress(x):
    '''
    x: b, f, t, 2
    '''
    # conduct sqrt power-compression
    mag, phase = torch.norm(x, dim=-1) ** 0.5, torch.atan2(x[..., -1], x[..., 0])
    return torch.stack((mag * torch.cos(phase), mag * torch.sin(phase)), dim=-1)


def stft_then_sqrt_comp(x, is_multi_channel, sr, win_size_sec, win_shift_sec, num_fft):
    '''
    Params:
        x: b, [m,] s
    Returns:
        y: b, [m,] f, t, 2
    '''
    x = x.unsqueeze(0)
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

    return y.squeeze(0)


def make_pathdirs(path):
    dirname = os.path.dirname(path)
    if os.path.exists(dirname):
        return
    os.makedirs(os.path.dirname(path))


def write_audio(path, data, sample_rate):
    """
    data: [n_channels,] n_samples
    """
    make_pathdirs(path)

    if len(data.shape) == 2:
        data = rearrange(data, "c s -> s c")

    sf.write(path, data, sample_rate)


def active_noise_rms(noise, fs, energy_thresh=-50):
    """Returns the clean and noise RMS of the noise calculated only in the active portions"""
    window_size = 100  # in ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    noise_active_segs = []

    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        noise_win = noise[sample_start:sample_end]
        noise_seg_rms = (noise_win**2).mean() ** 0.5
        # Considering frames with energy
        if noise_seg_rms > 10 ** (energy_thresh / 20):
            noise_active_segs = np.append(noise_active_segs, noise_win)
        sample_start += window_samples

    if len(noise_active_segs) != 0:
        noise_rms = (noise_active_segs**2).mean() ** 0.5
    else:
        noise_rms = EPS

    return noise_rms


def cal_coeff_for_adjusting_relative_energy(
    wav1: np.ndarray, wav2: np.ndarray, target_dB: float
):
    r"""calculate the coefficient used for adjusting the relative energy of two signals

    Args:
        wav1: the first wav
        wav2: the second wav
        target_dB: the target relative energy in dB, i.e. after adjusting: 10 * log_10 (average_energy(wav1) / average_energy(wav2 * coeff)) = target_dB

    Returns:
        float: coeff
    """
    # compute averaged energy over time and channel
    ae1 = np.sum(wav1**2) / np.prod(wav1.shape)
    ae2 = np.sum(wav2**2) / np.prod(wav2.shape)
    # compute the coefficients
    coeff = np.sqrt(ae1 / (ae2 + EPS) * np.power(10, -target_dB / 10))
    return coeff  # multiply it with wav2


def adjust_dbfs(audio, target_dbfs):
    rms = (audio**2).mean() ** 0.5
    scale = 10 ** (target_dbfs / 20) / (rms + EPS)
    return audio * scale


def bounded_scale(audio, max_abs=1):
    assert len(audio.shape) == 1, audio.shape
    max_v = np.max(np.abs(audio), axis=0)
    if max_v > max_abs:
        audio = audio / max_v
    return audio


def trim_silence(a, samplerate, threshold=0.01, frame_size=0.01):
    """
    从给定的音频数据中截取掉静音部分。

    参数:
    audio_data (numpy.ndarray): 一维 NumPy 数组，包含音频样本。
    samplerate (int): 音频的采样率。
    threshold (float): 静音阈值，低于此值的帧被认为是静音。
    frame_size (float): 帧的时间长度（秒）。
    hop_length (float): 帧之间的时间间隔（秒）。

    返回:
    numpy.ndarray: 截取后的音频数据。
    """
    assert len(a.shape) == 1, (
        a.shape,
        "multi channel audio trim is not implemented yet",
    )
    # 计算帧能量
    frame_len = int(samplerate * frame_size)
    # 计算填充后的总长度
    total_length = (len(a) + frame_len - 1) // frame_len * frame_len

    # 创建填充后的数组
    a = np.pad(a, (0, total_length - len(a)), mode="constant", constant_values=0)

    assert len(a) % frame_len == 0, (len(a), frame_len)

    a = a.reshape(-1, frame_len)

    u = a / (np.max(np.abs(a)) + 1e-10)
    e = np.sqrt(np.mean(u**2, axis=-1))
    # e = repeat(e,'n -> n m',m=frame_len)
    # print(e.shape)
    # return e.reshape(-1)

    indices = np.where(e >= threshold)[0]
    if len(indices) == 0:
        print(f"WARNING: trim silence failed {a.reshape(-1)}")
        return a.reshape(-1)
    start = indices[0]
    end = indices[-1] + 1
    assert end > start, (start, end)
    a = a[start:end].reshape(-1)
    return a


def crop_audio(audio, n_sample_points, random_crop, rng: np.random.Generator = None):
    """
    audio: (n,) or (n,c)
    """
    if n_sample_points <= 0:
        return audio
    m = n_sample_points
    while len(audio) < m:
        audio = np.concatenate([audio, audio], axis=0)
    n = len(audio)
    if random_crop:
        if rng is None:
            rng = np.random.default_rng()
        start = rng.integers(0, n - m + 1)
    else:
        start = 0
    return audio[start : start + m]


def load_and_crop(
    audio_path, n_sample_points, random_crop=True, rng: np.random.Generator = None
):
    audio, sr = sf.read(audio_path)
    return crop_audio(audio, n_sample_points, random_crop, rng)


def resample(audio, current_sr, target_sr):
    n = audio.shape[0]
    # print(audio.shape)
    if len(audio.shape) == 2:
        assert audio.shape[1] < 10, (
            "maybe (not) an error, check audio shape",
            audio.shape,
        )
    if current_sr != target_sr:
        audio = signal.resample(audio, int(n * target_sr / current_sr))
    return audio


def load_and_resample(path, resample_rs=16000):
    audio, sr = sf.read(path)
    return resample(audio, sr, resample_rs)


def recover_scale(
    preds: Tensor,
    mixture: Tensor,
    scale_src_together: bool,
    norm_if_exceed_1: bool = True,
) -> Tensor:
    """recover wav's original scale by solving min ||Y^T a - X||F, cuz sisdr will lose scale

    Args:
        preds: prediction, shape [batch, n_src, time]
        mixture: mixture or noisy or reverberant signal, shape [batch, time]
        scale_src_together: keep the relative ennergy level between sources. can be used for scale-invariant SA-SDR
        norm_max_if_exceed_1: norm the magitude if exceeds one

    Returns:
        Tensor: the scale-recovered preds
    """
    # TODO: add some kind of weighting mechanism to make the predicted scales more precise
    # recover wav's original scale. solve min ||Y^T a - X||F to obtain the scales of the predictions of speakers, cuz sisdr will lose scale
    if scale_src_together:
        a = torch.linalg.lstsq(
            preds.sum(dim=-2, keepdim=True).transpose(-1, -2), mixture.unsqueeze(-1)
        ).solution
    else:
        a = torch.linalg.lstsq(preds.transpose(-1, -2), mixture.unsqueeze(-1)).solution

    preds = preds * a

    if norm_if_exceed_1:
        # normalize the audios so that the maximum doesn't exceed 1
        max_vals = torch.max(torch.abs(preds), dim=-1).values
        norm = torch.where(max_vals > 1, max_vals, 1)
        preds = preds / norm.unsqueeze(-1)
    return preds


def make_sample(
    rirs, rirs_dp, target_speech, ref_mic, noise_list, snr, dbfs, mc_noise=None
):
    """
    rirs: (n_mic,(1+n_noise),?) rirs[:][0] must be for target speech and rirs[:][1:] for noises
    rirs_dp: (n_mic,1,?)
    """

    noise_list = [bounded_scale(adjust_dbfs(x, -20)) for x in noise_list]

    if rirs is not None:
        L = len(target_speech)
        for noise in noise_list:
            assert len(noise) == L, (len(noise), L)

        M = len(rirs)
        S = len(rirs[0])
        sources = [target_speech] + noise_list

        # the array that will receive all the signals
        premix_signals = np.zeros((S, M, L))

        # compute the signal at every microphone in the array
        for m in np.arange(M):
            for s in np.arange(S):
                sig = sources[s]
                h = rirs[m][s]
                premix_signals[s, m] += fftconvolve(h, sig)[:L]

        h_tgt_dp_ref = rirs_dp[ref_mic][0]
        clean_dp_ref = np.zeros(L)
        clean_dp_ref += fftconvolve(h_tgt_dp_ref, target_speech)[:L]
        target = clean_dp_ref

        if S > 1:
            rvbt_mixed_noise = np.sum(premix_signals[1:], axis=0)  # n_mics, len_seq
        else:
            rvbt_mixed_noise = np.zeros((M, L))

        if mc_noise is not None:
            rvbt_mixed_noise = rvbt_mixed_noise + rearrange(mc_noise, "n c -> c n")

        rvbt_speech = premix_signals[0]  # n_mics, len_seq
    else:
        assert target_speech.shape == mc_noise.shape, (
            target_speech.shape,
            mc_noise.shape,
        )
        target = target_speech[:, ref_mic]
        rvbt_speech = rearrange(target_speech, "n c -> c n")
        rvbt_mixed_noise = rearrange(mc_noise, "n c -> c n")

    scale_noise = cal_coeff_for_adjusting_relative_energy(
        rvbt_speech, rvbt_mixed_noise, snr
    )

    rvbt_mixed_noise *= scale_noise

    noisy = rvbt_mixed_noise + rvbt_speech

    rms_noisy = (noisy**2).mean() ** 0.5
    scale_noisy = 10 ** (dbfs / 20) / (rms_noisy + EPS)

    noisy *= scale_noisy
    target *= scale_noisy

    max_v = np.max(np.abs(noisy), axis=(0, 1))
    if max_v > 1:
        noisy = noisy / max_v
        target = target / max_v

    return {"noisy": noisy, "target": target}  # n_mics, n_samples  # n_samples


def cal_angle_deg_abs(v1, v2):
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def cal_angle_deg_ccw(v_from, v_to):
    abs_angle = cal_angle_deg_abs(v_from, v_to)
    is_clock_wise = (v_from[0] * v_to[1] - v_from[1] * v_to[0]) < 0
    deg = -abs_angle if is_clock_wise else abs_angle
    return deg


def rotate_matrix_ccw(deg):
    rad = np.radians(deg)
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    return R


def gen_random_config_value(config, rng: np.random.Generator):
    if config[0] == "continuous":
        assert len(config) == 3, config
        return rng.uniform(config[1], config[2])
    elif config[0] == "discrete":
        return rng.choice(config[1:])
    else:
        raise ValueError(config)


def generate_surround_noise_positions(
    room_dim, p_mics_center, p_target, opt, rng: np.random.Generator
):
    p_mics_center_2d = p_mics_center[:2]
    p_target_2d = p_target[:2]
    v = p_target_2d - p_mics_center_2d
    ox = np.array([1.0, 0.0])
    deg_target = cal_angle_deg_ccw(ox, v)
    deg_m = opt["min_deg_to_target"]
    n_noise = gen_random_config_value(opt["n"], rng)
    deg_sec = (360 - 2 * deg_m) / n_noise
    p_noise_list = []
    for i in range(n_noise):
        deg_start = deg_target + deg_m + i * deg_sec
        deg_end = deg_start + deg_sec
        for _ in range(10):
            deg_noise = rng.uniform(deg_start, deg_end)
            rad_noise = np.deg2rad(deg_noise)
            dist_noise = gen_random_config_value(opt["dist_to_mic"], rng)
            x_noise = p_mics_center_2d[0] + dist_noise * np.cos(rad_noise)
            y_noise = p_mics_center_2d[1] + dist_noise * np.sin(rad_noise)
            if (
                opt["min_dist_to_wall"]
                < x_noise
                < room_dim[0] - opt["min_dist_to_wall"]
            ) and (
                opt["min_dist_to_wall"]
                < y_noise
                < room_dim[1] - opt["min_dist_to_wall"]
            ):
                break
            x_noise = None
            y_noise = None
        if x_noise != None and y_noise != None:
            p_noise_list.append(
                (x_noise, y_noise, gen_random_config_value(opt["h"], rng))
            )
    return p_noise_list


def generate_random_rirs(opt, rng: np.random.Generator, specific=None):
    """
    specific keys:
        - room_dim : [a,b,c]
        - target_xyz : [x,y,z]
        - mics_xyz : [x,y,z]
        - noise_xyz_list : [[x,y,z], ...]
        - rt60 : x
    """

    if opt["target"]["type"] == "real" and opt["noise"]["type"] == "real":
        return {
            "rirs": {
                "rirs": None,
                "rirs_dp": None,
                "info": None,
            },
            "room": None,
        }

    if specific is None:
        specific = dict()

    # generate random room

    if "room_dim" in specific:
        room_dim = np.array(specific["room_dim"])
    else:
        min_dim = np.array(opt["room"]["min_dim"])
        max_dim = np.array(opt["room"]["max_dim"])
        room_dim = min_dim + (max_dim - min_dim) * rng.random([3])

    # load mics

    p_mics = np.array([[mic["x"], mic["y"]] for mic in opt["mic_array"]["mics"]])  # 8,2
    p_mics = p_mics.T  # 2, n_mics
    direction_mics = opt["mic_array"]["direction"]
    direction_mics = np.array([direction_mics["x"], direction_mics["y"]])  # 2

    # generate random target and mics position

    fail_count = 0

    random_target = True
    random_mics = True

    if "target_xyz" in specific:
        target_x, target_y, target_z = specific["target_xyz"]
        random_target = False

    if "mics_xyz" in specific:
        mics_x, mics_y, mics_z = specific["mics_xyz"]
        random_mics = False

    if random_target or random_mics:
        while True:

            if random_mics:
                d = opt["mic_array"]["min_dist_to_wall"]
                mics_x = rng.uniform(d, room_dim[0] - d)
                mics_y = rng.uniform(d, room_dim[1] - d)
                mics_z = gen_random_config_value(opt["mic_array"]["h"], rng)

            if random_target:
                rad_to_mic = rng.uniform(0, 2 * np.pi)
                dist_to_mic = gen_random_config_value(opt["target"]["dist_to_mic"], rng)
                target_x = mics_x + dist_to_mic * np.cos(rad_to_mic)
                target_y = mics_y + dist_to_mic * np.sin(rad_to_mic)
                d = opt["target"]["min_dist_to_wall"]
                if not (
                    d < target_x < room_dim[0] - d and d < target_y < room_dim[1] - d
                ):
                    fail_count += 1
                    continue
                target_z = gen_random_config_value(opt["target"]["h"], rng)

            break

    p_target = np.array([target_x, target_y, target_z])
    p_target_2d = p_target[:2]
    p_mics_cen = np.array([mics_x, mics_y, mics_z])
    p_mics_cen_2d = p_mics_cen[:2]

    # rotate the mic array to make its direction is toward the target
    assert opt["target"]["fixed_doa"], "Not supported"
    direction_target = p_target_2d - p_mics_cen_2d
    deg = cal_angle_deg_ccw(direction_mics, direction_target)
    deg_jitter = gen_random_config_value(opt["target"]["doa_jitter"], rng)
    R = rotate_matrix_ccw(deg + deg_jitter)
    direction_mics = np.dot(R, direction_mics)
    p_mics = np.dot(R, p_mics)  # 2, n_mics
    p_mics = np.concatenate([p_mics, np.zeros((1, p_mics.shape[1]))], 0)  # 3, n_mics
    p_mics = p_mics + p_mics_cen.reshape((3, 1))

    # generate random noises position

    if "noise_xyz_list" in specific:
        p_noise_list = specific["noise_xyz_list"]
    else:
        noise_type = opt["noise"]["type"]
        if noise_type == "surround":
            n_try = 0
            while True:
                n_try += 1
                p_noise_list = generate_surround_noise_positions(
                    room_dim, p_mics_cen, p_target, opt["noise"], rng
                )
                if len(p_noise_list) > 0:
                    break
                if n_try > 5:
                    print("WARNING: sample generation meets difficulty, try to fix it")
        elif noise_type == "real":
            p_noise_list = []
        else:
            raise NotImplementedError

    # generate random rt60

    if "rt60" in specific:
        rt60_tgt = specific["rt60"]
        e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    else:
        while True:
            rt60_tgt = gen_random_config_value(opt["room"]["rt60"], rng)
            try:
                e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
            except ValueError:
                # room too large for given rt60
                fail_count += 1
                continue
            break

    if fail_count >= 50:
        print(
            f"Random position generation failed {fail_count} times in a sample, the restriction may be too tight"
        )

    fs = opt["audio"]["fs"]

    rir_method = opt["rir_method"]

    if rir_method == "ism":
        room = pra.ShoeBox(
            room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
        )
    elif rir_method == "hybrid":
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.Material(e_absorption),
            max_order=3,
            ray_tracing=True,
            air_absorption=True,
        )
    else:
        raise ValueError(rir_method)

    # always add p_target at idx 0
    room.add_source(p_target)

    for p in p_noise_list:
        room.add_source(p)

    room.add_microphone_array(p_mics)
    room.compute_rir()
    freefield = pra.AnechoicRoom(3, fs=fs)
    freefield.add_source(p_target)
    freefield.add_microphone_array(p_mics)
    freefield.compute_rir()

    return {
        "rirs": {
            "rirs": room.rir,  # n_mics, n_sources, len_seq
            "rirs_dp": freefield.rir,
            "info": {
                "room_dim": room_dim,
                "p_target": p_target,
                "p_noise_list": p_noise_list,
                "p_mics": p_mics,
                "rt60": rt60_tgt,
                "rir_method": rir_method,
                "fs": fs,
            },
        },
        "room": room,
    }


def sample_speech(speech_list, bias, n_sample_points, real_speech_converter, opt, rng):
    sr = opt["audio"]["fs"]
    speech_audio_list = []
    total_len = 0
    N_TRIES = 100
    for _ in range(N_TRIES):
        i = (rng.integers(len(speech_list)) + bias) % len(speech_list)
        speech = speech_list[i]
        speech_path = speech
        speech_audio = load_and_resample(speech_path, sr)  # (n,) or (n,c)
        if speech_audio.shape[0] == 0:
            # print(f'ATTENTION: audio {speech_path} has 0 length, skip it')
            continue
        is_mc_audio = len(speech_audio.shape) == 2

        if opt["target"]["type"] == "real":
            speech_audio = real_speech_converter(speech_audio)

        if n_sample_points <= 0:
            # no clip
            speech_audio_list.append(speech_audio)
            break

        if not is_mc_audio:
            # multi channel audio trim is not implemented yet
            speech_audio = trim_silence(speech_audio, sr)

        total_len += speech_audio.shape[0]
        speech_audio_list.append(speech_audio)
        if total_len >= n_sample_points:
            break

    assert (
        total_len >= n_sample_points
    ), f"N_TRIES={N_TRIES} may be too small? {[len(a) for a in speech_audio]}"
    speech_audio = np.concatenate(speech_audio_list, 0)
    if len(speech_audio_list) > 1:
        k = rng.integers(0, len(speech_audio) - 1)
        speech_audio = np.roll(speech_audio, k, 0)

    return speech_audio


def generate_random_sample(
    opt,
    rng: np.random.Generator,
    speech_list,
    noise_list,
    clip_seconds,
    specific=None,
    real_noise_converter=None,
    real_speech_converter=None,
):

    if specific is None:
        specific = dict()

    if "rirs" in specific:
        rirs_pack = specific["rirs"]
        room = None
    else:
        output = generate_random_rirs(opt, rng, specific)
        rirs_pack = output["rirs"]
        room = output["room"]

    rirs = rirs_pack["rirs"]
    rirs_dp = rirs_pack["rirs_dp"]

    sr = 16000
    n_sample_points = sr * clip_seconds
    if "speech_audio" in specific:
        speech_audio = specific["speech_audio"]
    else:
        speech_audio = sample_speech(
            speech_list, 0, n_sample_points, real_speech_converter, opt, rng
        )

    if n_sample_points == 0:
        n_sample_points = len(speech_audio)

    if opt["noise"]["type"] == "real":
        n_noise = 1
    else:
        n_noise = len(rirs[0]) - 1

    use_noise_list = rng.choice(noise_list, n_noise)

    random_crop = (
        False
        if ("random_crop" in specific and specific["random_crop"] == False)
        else True
    )

    audios = [
        crop_audio(speech_audio, n_sample_points, random_crop=random_crop, rng=rng)
    ]

    mc_noise = None
    for use_noise in use_noise_list:
        if opt["noise"]["type"] == "real":
            mc_noise = real_noise_converter(load_and_crop(use_noise, n_sample_points))
        else:
            audios.append(
                load_and_crop(
                    use_noise, n_sample_points, random_crop=random_crop, rng=rng
                )
            )

    ref_mic = opt["mic_array"]["ref_mic"]

    if "snr" in specific:
        snr = specific["snr"]
    else:
        snr = gen_random_config_value(opt["snr"], rng)

    if "noisy_dBFS" in specific:
        dbfs = specific["noisy_dBFS"]
    else:
        dbfs = gen_random_config_value(opt["noisy_dBFS"], rng)

    sample = make_sample(
        rirs, rirs_dp, audios[0], ref_mic, audios[1:], snr, dbfs, mc_noise=mc_noise
    )
    sample["snr"] = snr
    sample["dbfs"] = dbfs
    sample["room"] = room
    return sample


def read_list(list_file):
    return list(map(str, Path(list_file).rglob("*.wav", recurse_symlinks=True)))


# def worker(rank, world_size, queue, opt, output_path, max_size=4 * 1024**3):
#     with open(opt["mcse_settings"], "r") as f:
#         mcse_settings = json.load(f)

#     speech_list = read_list(opt["speech_list"])
#     noise_list = read_list(opt["noise_list"])
#     clip_seconds = opt["clip_seconds"]
#     sr = mcse_settings["audio"]["fs"]
#     rng = np.random.default_rng()

#     output_path = os.path.join(output_path, f"shards_00_{rank:02d}_%04d.tar")

#     with wds.ShardWriter(output_path, maxsize=max_size) as sink:
#         for i in range(rank, len(speech_list), world_size):
#             speech_audio = sample_speech(
#                 speech_list, i, sr * clip_seconds, None, mcse_settings, rng
#             )

#             sample = generate_random_sample(
#                 mcse_settings,
#                 rng,
#                 speech_list,
#                 noise_list,
#                 clip_seconds,
#                 specific={"speech_audio": speech_audio},
#             )
#             target = torch.tensor(sample["target"], dtype=torch.float32)
#             noisy = torch.tensor(sample["noisy"], dtype=torch.float32)
#             sample = {"noisy.pth": noisy, "target.pth": target, "__key__": f"sample_{i}"}
#             sink.write(sample)
#             queue.put(1)


# def create_webdataset(opt, output_path, max_size=4 * 1024**3):

#     Path(output_path).parent.mkdir(exist_ok=True, parents=True)
#     queue = mp.Queue()

#     process: list[mp.Process] = []
#     num_workers = 16
#     for i in range(num_workers):
#         p = mp.Process(
#             target=worker, args=(i, num_workers, queue, opt, output_path, max_size)
#         )
#         p.start()
#         process.append(p)
#     total_sample = len(read_list(opt["speech_list"]))
#     p_bar = tqdm(total=total_sample)
#     cnt = 0
#     while cnt <= total_sample:
#         _ = queue.get()
#         cnt += 1
#         p_bar.update(1)

#     for p in process:
#         p.join()


# def worker(speech_list, noise_list, i, sr, clip_seconds, mcse_settings, rng):
#     speech_audio = sample_speech(
#         speech_list, i, sr * clip_seconds, None, mcse_settings, rng
#     )

#     sample = generate_random_sample(
#         mcse_settings,
#         rng,
#         speech_list,
#         noise_list,
#         clip_seconds,
#         specific={"speech_audio": speech_audio},
#     )
#     target = torch.tensor(sample["target"], dtype=torch.float32)
#     noisy = torch.tensor(sample["noisy"], dtype=torch.float32)
#     return {"noisy.pth": noisy, "target.pth": target, "__key__": f"s_{i}"}


# def create_webdataset(opt, output_path, max_size=4 * 1024**3):

#     Path(output_path).parent.mkdir(exist_ok=True, parents=True)

#     with open(opt["mcse_settings"], "r") as f:
#         mcse_settings = json.load(f)

#     speech_list = read_list(opt["speech_list"])
#     noise_list = read_list(opt["noise_list"])
#     clip_seconds = opt["clip_seconds"]
#     sr = mcse_settings["audio"]["fs"]
#     rng = np.random.default_rng()

#     with wds.ShardWriter(output_path, maxsize=max_size) as sink:
#         with ProcessPoolExecutor(max_workers=16) as executor:
#             futures = []
#             for i in range(len(speech_list)):
#                 futures.append(
#                     executor.submit(
#                         worker,
#                         speech_list,
#                         noise_list,
#                         i,
#                         sr,
#                         clip_seconds,
#                         mcse_settings,
#                         rng,
#                     )
#                 )
#             for future in tqdm(as_completed(futures), total=len(speech_list)):
#                 try:
#                     result = future.result()
#                     sink.write(result)
#                 except Exception as e:
#                     print(e)


def worker(rank, world_size, queue, opt):
    with open(opt["mcse_settings"], "r") as f:
        mcse_settings = json.load(f)

    speech_list = read_list(opt["speech_list"])
    noise_list = read_list(opt["noise_list"])
    clip_seconds = opt["clip_seconds"]
    sr = mcse_settings["audio"]["fs"]
    rng = np.random.default_rng()

    for i in range(rank, len(speech_list), world_size):
        speech_audio = sample_speech(
            speech_list, i, sr * clip_seconds, None, mcse_settings, rng
        )

        sample = generate_random_sample(
            mcse_settings,
            rng,
            speech_list,
            noise_list,
            clip_seconds,
            specific={"speech_audio": speech_audio},
        )

        stft_params = {
            "sr": 16000,
            "win_size_sec": 0.02,
            "win_shift_sec": 0.01,
            "num_fft": 320,
        }

        target = torch.tensor(sample["target"], dtype=torch.float32)
        target = stft_then_sqrt_comp(target, False, **stft_params)
        noisy = torch.tensor(sample["noisy"], dtype=torch.float32)
        noisy = stft_then_sqrt_comp(noisy, True, **stft_params)
        sample = {"noisy.pth": noisy, "target.pth": target, "__key__": f"s_{i}"}

        queue.put(sample)


def create_dataset(
    opt, output_path, num_workers=32, max_size=10000 * 1024**3, maxcount=1000
):

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    queue = mp.SimpleQueue()

    process: list[mp.Process] = []
    for i in range(num_workers):
        p = mp.Process(target=worker, args=(i, num_workers, queue, opt))
        p.start()
        process.append(p)
    total_sample = len(read_list(opt["speech_list"]))
    p_bar = tqdm(total=total_sample)
    cnt = 0

    with wds.ShardWriter(output_path, maxsize=max_size, maxcount=maxcount) as sink:
        while cnt < total_sample:
            sample = queue.get()
            sink.write(sample)
            cnt += 1
            p_bar.update(1)

    for p in process:
        p.join()
    p_bar.close()


if __name__ == "__main__":
    # opt = {
    #     "speech_list": "/home/liuxin/datasets/chinese_speech/commonvoice",
    #     "noise_list": "/data1/liuxin/datasets/single_noise/dns_noise_16khz",
    #     "mcse_settings": "/home/liuxin/mcse/dataset/mcse_settings/v13.json",
    #     "clip_seconds": 12,
    # }
    opt = {
        "speech_list": "/home/liuxin/datasets/chinese_speech/commonvoice",
        "noise_list": "/data1/liuxin/datasets/single_noise/dns_noise_16khz",
        "mcse_settings": "/home/liuxin/mcse/dataset/mcse_settings/v13.json",
        "clip_seconds": 8,
    }
    print(len(read_list(opt["speech_list"])))
    print(len(read_list(opt["noise_list"])))
    create_dataset(opt, "/data1/liuxin/datasets/audio_test/dataset-%05d.tar")
