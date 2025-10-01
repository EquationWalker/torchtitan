import torch
from torchtitan.tools import utils
from .. import eabnet_qwen3_configs
from ..model.model import EaBNetQwen3
import argparse
import json
from ..utils import make_converter
from pathlib import Path
import numpy as np
from ..utils import sqrt_decomp_then_istft, stft_then_sqrt_comp
from functools import partial
from einops import rearrange
from moviepy import VideoFileClip, AudioFileClip
import soundfile as sf
import numpy as np
import torch.multiprocessing as mp
from scipy.signal import resample

def inference_sliding_window1(
    inference_fn,
    target_sr,
    converter,
    noisy_path,
    output_path,
    chunk_size_second=12,
    overlap_second=6,
    verbose=True,
):
    """
    Streaming sliding-window inference with cross-fade in overlap regions.

    Params:
      inference_fn: callable(noisy_np (n,c) cpu) -> esti (n,)
      converter: callable to convert raw waveform to model input (keeps (n,c))
    """
    noisy_path = str(noisy_path)
    output_path = str(output_path)

    with sf.SoundFile(noisy_path, "r") as sf_noisy:
        noisy_sr = sf_noisy.samplerate
        total_samples = int(sf_noisy.frames * target_sr / noisy_sr)
        if verbose:
            print(f"[I] {noisy_path} sr={noisy_sr}, duration={total_samples/target_sr:.1f}s -> target_sr={target_sr}")

        chunk_samples = int(chunk_size_second * target_sr)
        overlap_samples = int(overlap_second * target_sr)
        hop = chunk_samples - overlap_samples
        if hop <= 0:
            raise ValueError("chunk_size_second must be larger than overlap_second")

        # open output file for streaming write
        sf_out = sf.SoundFile(output_path, mode="w", samplerate=target_sr, channels=1, subtype="PCM_16")
        last_overlap = None  # buffer for last overlap samples (1D numpy) to blend with next chunk

        si = 0
        chunk_idx = 0
        try:
            while si < total_samples:
                ei = min(si + chunk_samples, total_samples)
                desired_len = ei - si  # in target_sr samples

                # read from source (frame count in source sr)
                sf_noisy.seek(int(si * noisy_sr / target_sr))
                read_frames = int(np.ceil(desired_len * noisy_sr / target_sr))
                noisy_chunk = sf_noisy.read(read_frames, dtype="float32")
                if noisy_chunk is None or len(noisy_chunk) == 0:
                    # safety break
                    break

                # if mono, ensure shape (N,1)
                noisy_chunk = np.asarray(noisy_chunk, dtype=np.float32)
                if noisy_chunk.ndim == 1:
                    noisy_chunk = noisy_chunk[:, None]

                # resample to exact desired_len
                if noisy_chunk.shape[0] != desired_len:
                    noisy_chunk = resample(noisy_chunk, desired_len, axis=0)

                # pad if last chunk shorter than chunk_samples
                padded = False
                if noisy_chunk.shape[0] < chunk_samples:
                    pad_n = chunk_samples - noisy_chunk.shape[0]
                    noisy_chunk = np.pad(noisy_chunk, ((0, pad_n), (0, 0)), mode="constant")
                    padded = True

                # converter + inference
                model_in = converter(noisy_chunk)  # expect (n, c)
                esti = inference_fn(model_in)  # (n,)
                # trim padding if any
                esti = esti[:desired_len]

                # blending / writing:
                # if first chunk: write full esti
                if last_overlap is None:
                    # if chunk larger than desired_len (shouldn't be), trim
                    sf_out.write(esti)
                    # keep last overlap portion for next chunk if there is overlap
                    if overlap_samples > 0:
                        last_overlap = esti[-overlap_samples:].copy()
                else:
                    # Blend last_overlap and current chunk's first overlap_samples via linear crossfade
                    cur_front = esti[:overlap_samples]
                    # If current chunk smaller than overlap (very short), handle gracefully
                    if len(cur_front) < overlap_samples:
                        # blend as much as possible then write remainder
                        L = len(cur_front)
                        fade_in = np.linspace(0.0, 1.0, L, endpoint=True)
                        fade_out = 1.0 - fade_in
                        blended = last_overlap[-L:] * fade_out + cur_front * fade_in
                        # overwrite the last overlap in file: we cannot seek-back easily in streaming write,
                        # so instead store output as follows: we will maintain a small buffer of last written non-overlap region.
                        # To simplify: we will write the blended part first (since last_overlap was not yet flushed beyond overlap region)
                        # Implementation simplification: reopen file in 'r+' would be complex; instead we'll maintain an internal buffer
                        # Simpler and robust approach: write blended part, then write the remainder of current chunk after overlap.
                        sf_out.write(blended)
                        if desired_len > L:
                            sf_out.write(esti[L:])
                        # update last_overlap: take last overlap_samples from this chunk if possible
                        last_overlap = esti[-overlap_samples:].copy() if len(esti) >= overlap_samples else esti.copy()
                    else:
                        fade_in = np.linspace(0.0, 1.0, overlap_samples, endpoint=True)
                        fade_out = 1.0 - fade_in
                        blended = last_overlap * fade_out + cur_front * fade_in
                        # write blended overlap
                        sf_out.write(blended)
                        # write the rest of esti after overlap
                        if desired_len > overlap_samples:
                            sf_out.write(esti[overlap_samples:])
                        # update last_overlap for next round
                        if desired_len >= overlap_samples:
                            last_overlap = esti[-overlap_samples:].copy()
                        else:
                            last_overlap = esti.copy()

                chunk_idx += 1
                si += hop
                if verbose:
                    percent = 100.0 * min(ei, total_samples) / total_samples
                    print(f"\rProcessed {chunk_idx} chunks, progress: {percent:.1f}% ({si}/{total_samples})", end="")
        finally:
            sf_out.close()
            if verbose:
                print("\nFinished streaming write to", output_path)

def inference_sliding_window(
    inference_fn,
    target_sr,
    converter,
    noisy_path,
    output_path,
    chunk_size_second=12,
    overlap_second=6,
    verbose=True,
):
    """
    Args:
        inference_fn: callable, (n, c) -> (n,)
        target_sr: int, 输出采样率
        converter: callable, 将波形转换为模型需要的输入格式
        noisy_path: str, 输入音频路径
        output_path: str, 输出音频路径
        chunk_size_second: float, 每个切片长度（秒）
        overlap_second: float, 切片重叠长度（秒）
        verbose: bool, 是否打印进度
    """
    with sf.SoundFile(noisy_path, "r") as sf_noisy:
        noisy_sr = sf_noisy.samplerate
        total_samples = int(sf_noisy.frames * target_sr / noisy_sr)

        if verbose:
            print(f"original sr = {noisy_sr}, duration = {total_samples / target_sr:.1f} s")

        chunk_size = int(chunk_size_second * target_sr)
        overlap_size = int(overlap_second * target_sr)

        # 输出缓冲区
        output = np.zeros(total_samples, dtype=np.float32)

        si = 0
        while si < total_samples:
            ei = min(si + chunk_size, total_samples)

            # 读取并重采样
            sf_noisy.seek(int(si * noisy_sr / target_sr))
            noisy = sf_noisy.read(int((ei - si) * noisy_sr / target_sr), dtype="float32")
            noisy = resample(noisy, ei - si, axis=0)

            # padding
            if len(noisy) < chunk_size:
                noisy = np.pad(noisy, ((0, chunk_size - len(noisy)), (0, 0)))

            # 转换 & 推理
            noisy = converter(noisy)
            esti = inference_fn(noisy)

            # 去掉 pad
            esti = esti[: ei - si]

            # overlap 区域平滑拼接：前半段保留已有结果，后半段覆盖
            start = si if si == 0 else si + overlap_size // 2
            output[start:ei] = esti[start - si :]

            if verbose:
                print(f"\r{100 * ei / total_samples:.1f}% [{si},{ei})", end="")

            si += chunk_size - overlap_size

    # 一次性写入
    sf.write(output_path, output, target_sr)

    if verbose:
        print("\nDone!")

@torch.inference_mode()
def load_model(flavor="1.4B", device="cuda", ckpt_path=None, dtype=torch.float32):
    with (
        torch.device("meta"),
        utils.set_default_dtype(dtype),
    ):
        model = EaBNetQwen3(eabnet_qwen3_configs[flavor])
        
    model.to_empty(device=torch.device(device))
    model.init_weights(buffer_device=None)
    if ckpt_path is not None:
        print("Loading checkpoint")
        sd = torch.load(ckpt_path, map_location=torch.device("cpu"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Missing key: {missing}\n")
        print(f"Unexpected key: {unexpected}\n")
    return model

@torch.inference_mode()
def _inference_fn(model_cuda, noisy_np_cpu):
    """
    noisy_np_cpu: (n,c)
    Returns: (n,)
    """
    noisy_np_cpu = rearrange(noisy_np_cpu, "n c->c n")

    stft_params = {
        "sr": 16000,
        "win_size_sec": 0.02,
        "win_shift_sec": 0.01,
        "num_fft": 320,
    }

    x = torch.Tensor(noisy_np_cpu)[None].cuda()
    x_stft = stft_then_sqrt_comp(x, is_multi_channel=True, **stft_params)
    y_stft = model_cuda(x_stft)
    y = sqrt_decomp_then_istft(y_stft, is_multi_channel=False, **stft_params)
    res = y[0].cpu().numpy()
    return res


def prepare_model(rank, args):
    print(f"Rank-{rank} loading checkpoint from {args.ckpt_path}")
    model = load_model(ckpt_path=args.ckpt_path)
    return model

def prepare_converter(mcse_settings, device_positions):
    if device_positions is None:
        print("warning: no device positions file specified")
        return lambda x: x
    with open(device_positions, "r") as f:
        device_mics = json.load(f)
        return make_converter("device to model", mcse_settings, device_mics)


def batch_inference(
    rank,
    num_gpus,
    num_proc,
    wav_list,
    args
):

    torch.cuda.set_device(rank % num_gpus)
    model = prepare_model(rank, args)

    converter = prepare_converter(args.mcse_settings, args.device_positions)

    with open(args.mcse_settings, "r") as f:
        opt = json.load(f)
        target_sr = opt["audio"]["fs"]

    noisy_dir = Path(args.noisy_dir)
    output_dir = Path(args.output_dir)

    inference_fn = partial(_inference_fn, model)

    for i in range(rank, len(wav_list), num_proc):
        wav_file = wav_list[i]
        noisy_path = wav_file
        tmp_out_path = output_dir / wav_file.relative_to(noisy_dir).parent
        tmp_out_path.mkdir(exist_ok=True, parents=True)
        save_audio_path = tmp_out_path / f"{noisy_path.stem}.wav"
        if save_audio_path.exists():
            continue

        print(f"Rank-{rank} processing {noisy_path} ({i}/{len(wav_list)})")
        print(f"Rank-{rank} writing to {save_audio_path}")
        inference_sliding_window(
            inference_fn,
            target_sr,
            converter,
            noisy_path,
            save_audio_path,
            verbose=True,
        )

        video_path = noisy_path.parent / f"{noisy_path.stem}.mp4"
        if video_path.exists():
            save_video_path = tmp_out_path / f"{noisy_path.stem}.mp4"
            video = VideoFileClip(str(video_path))
            audio = AudioFileClip(str(save_audio_path))
            video_with_audio = video.with_audio(audio)
            video_with_audio.write_videofile(
                str(save_video_path), codec="libx264", audio_codec="aac"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    # model related
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--mcse_settings", type=str)
    # dataset related
    parser.add_argument("--noisy_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--device_positions", type=str, default=None)
    parser.add_argument("--n_workers", type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_args()
    print(args)
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"making dir {output_dir}")
        output_dir.mkdir(parents=True)
        
    noisy_dir = Path(args.noisy_dir)
    wav_list = [
        wav_file
        for wav_file in noisy_dir.rglob("*.wav")
        if wav_file.stem[-8:] != "channel1"
    ]
    print(f"Total number of audio files:{len(wav_list)}")

    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    num_proc = args.n_workers  # min(num_gpus,len(wav_list))

    print(f"Running {num_proc} processes.")

    # 启动多个进程
    processes = []
    for rank in range(num_proc):
        p = mp.Process(
            target=batch_inference,
            args=(
                rank,
                num_gpus,
                num_proc,
                wav_list,
                args
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("All processes have finished.")
