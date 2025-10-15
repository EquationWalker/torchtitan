cd ..
# python -m torchtitan.experiments.eabnet_qwen3.scripts.inference \
#     --ckpt_path "/data2/liuxin/expr/outputs/torch_pth/200000.pth" \
#     --mcse_settings "torchtitan/experiments/eabnet_qwen3/scripts/1.json" \
#     --n_workers 4 \
#     --noisy_dir '/data1/liuxin/datasets/mc_rebuttal/val_real/noisy' \
#     --output_dir '/data1/liuxin/datasets/mc_rebuttal/val_real/200K_eabQwen3_overlap3/pred' \
#     --device_positions '/home/liuxin/code/MCAudioEnhanceServing-master/src/evaluate/device_positions/config.json' \

# python -m torchtitan.experiments.eabnet_qwen3.scripts.inference \
#     --ckpt_path "/data2/liuxin/expr/outputs/torch_pth/90000.pth" \
#     --mcse_settings "torchtitan/experiments/eabnet_qwen3/scripts/1.json" \
#     --n_workers 4 \
#     --noisy_dir '/data1/liuxin/datasets/mc_rebuttal/val_real/noisy' \
#     --output_dir '/data1/liuxin/datasets/mc_rebuttal/val_real/90K_eabQwen3' \
#     --device_positions '/home/liuxin/code/MCAudioEnhanceServing-master/src/evaluate/device_positions/config.json' \

# python -m torchtitan.experiments.eabnet_qwen3.scripts.inference \
#     --ckpt_path "/data2/liuxin/expr/qwen_raw_loss_mask/torch_pth/480000.pth" \
#     --mcse_settings "torchtitan/experiments/eabnet_qwen3/scripts/1.json" \
#     --n_workers 4 \
#     --noisy_dir '/data1/liuxin/datasets/mc_rebuttal/val_real/noisy' \
#     --output_dir '/data1/liuxin/datasets/mc_rebuttal/val_real/480K_eabQwen3_raw_mask' \
#     --device_positions '/home/liuxin/code/MCAudioEnhanceServing-master/src/evaluate/device_positions/config.json' \

# python -m torchtitan.experiments.eabnet_qwen3.scripts.inference \
#     --ckpt_path "/data2/liuxin/expr/qwen_raw_loss_mask/torch_pth/540000.pth" \
#     --mcse_settings "torchtitan/experiments/eabnet_qwen3/scripts/1.json" \
#     --n_workers 4 \
#     --noisy_dir '/data1/liuxin/datasets/mc_rebuttal/val_real/noisy' \
#     --output_dir '/data1/liuxin/datasets/mc_rebuttal/val_real/540K_eabQwen3_raw_mask' \
#     --device_positions '/home/liuxin/code/MCAudioEnhanceServing-master/src/evaluate/device_positions/config.json' \

python -m torchtitan.experiments.eabnet_qwen3.scripts.inference \
    --ckpt_path "/data2/liuxin/expr/qwen_raw_loss_mask/torch_pth/590000.pth" \
    --mcse_settings "torchtitan/experiments/eabnet_qwen3/scripts/1.json" \
    --n_workers 4 \
    --noisy_dir '/data1/liuxin/datasets/mc_rebuttal/val_real/noisy' \
    --output_dir '/data1/liuxin/datasets/mc_rebuttal/val_real/590K_eabQwen3_raw_mask' \
    --device_positions '/home/liuxin/code/MCAudioEnhanceServing-master/src/evaluate/device_positions/config.json' \

