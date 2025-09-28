# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


def parallelize_eabnet(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    # if job_config.activation_checkpoint.mode != "none":
    #     apply_ac(model, job_config.activation_checkpoint)

    if parallel_dims.dp_shard_enabled:  # apply FSDP or HSDP
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard")
        else:
            dp_mesh_dim_names = ("dp_shard",)
            
        # model = torch.compile(model)
        # logger.info("Compiling with torch.compile")

        apply_fsdp(
            model,
            parallel_dims.world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            cpu_offload=job_config.training.enable_cpu_offload,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")
    model = torch.compile(model)
    logger.info("Compiling with torch.compile") 

    return model

from ..model.model import EaBNet
def apply_fsdp(
    model: nn.Module | EaBNet,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        cpu_offload (bool): Whether to offload model parameters to CPU. Defaults to False.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    encoder = model.en
    for i in encoder.meta_unet_list:
        # fully_shard(i.in_conv, **fsdp_config)
        # for j in i.enco:
        #     fully_shard(j, **fsdp_config)
        # for j in i.deco:
        #     fully_shard(j, **fsdp_config)
        fully_shard(i, **fsdp_config)
    fully_shard(encoder.last_conv, **fsdp_config)
    decoder = model.de
    for i in decoder.meta_unet_list:
        # for j in i.enco:
        #     fully_shard(j, **fsdp_config)
        # for j in i.deco:
        #     fully_shard(j, **fsdp_config)
        # fully_shard(i.last_conv, **fsdp_config)
        fully_shard(i, **fsdp_config)
    fully_shard(decoder.last_conv, **fsdp_config)
        
        
    for i in model.stcns:
        # for j in i.tcm_list:
        #     fully_shard(j, **fsdp_config)
        fully_shard(i, **fsdp_config)
            
    
    for rnn in model.bf_map.layers:
        fully_shard(rnn, **fsdp_config)
    fully_shard([model.bf_map.norm, model.bf_map.w_dnn], **fsdp_config, reshard_after_forward=False)
            

    # Wrap all the rest of model
    fully_shard(model, **fsdp_config)
    
    
def apply_compile(model: nn.Module | EaBNet):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # NOTE: This flag is needed for torch.compile to avoid graph breaking on dynamic shapes in token-choice MoE
    # but it is experimental.
    # torch._dynamo.config.capture_scalar_outputs = True
    
    for layer_id, transformer_block in model.bf_map.layers.named_children():
        # TODO: remove when torch.compile supports fullgraph=True for MoE
        fullgraph = True
        transformer_block = torch.compile(transformer_block, fullgraph=fullgraph)
        model.bf_map.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")


# def apply_ac(model: nn.Module, ac_config):
#     """Apply activation checkpointing to the model."""

#     for layer_id, block in model.double_blocks.named_children():
#         block = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
#         model.double_blocks.register_module(layer_id, block)

#     for layer_id, block in model.single_blocks.named_children():
#         block = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
#         model.single_blocks.register_module(layer_id, block)

#     logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")

