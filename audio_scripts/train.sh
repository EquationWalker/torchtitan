#!/usr/bin/bash
cd ..

PYTORCH_ALLOC_CONF="expandable_segments:True" 
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export HF_DATASETS_IN_MEMORY_MAX_SIZE=150994944000
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 -m torchtitan.experiments.eabnet_qwen3.train --job.config_file "torchtitan/experiments/eabnet_qwen3/train_configs/1_4B.toml"