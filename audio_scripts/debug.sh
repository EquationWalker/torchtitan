#!/usr/bin/bash
cd ..

PYTORCH_ALLOC_CONF="expandable_segments:True" 
export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 -m torchtitan.experiments.eabnet_qwen3.train --job.config_file "torchtitan/experiments/eabnet_qwen3/train_configs/debug.toml"


# PYTORCH_ALLOC_CONF="expandable_segments:True" 
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCHDYNAMO_VERBOSE=1
# # CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun --standalone --nproc_per_node=8 -m torchtitan.experiments.eabnet.train --job.config_file "torchtitan/experiments/eabnet/train_configs/debug_model.toml"