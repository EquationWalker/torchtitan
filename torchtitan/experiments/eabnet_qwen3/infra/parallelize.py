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
from ..model.model import EaBNetQwen3
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
    apply_compile(model)
    logger.info("Compiling with torch.compile") 

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
            
            
    # model = torch.compile(model)
    # logger.info("Compiling with torch.compile") 

    return model

def apply_fsdp(
    model: nn.Module | EaBNetQwen3,
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

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config)

    for layer_id, transformer_block in model.layers.items():
        fully_shard(
            transformer_block,
            **fsdp_config)

    # As an optimization, do not reshard_after_forward the last layers by default
    # since FSDP would prefetch them immediately after the forward pass
    if model.norm is not None and model.output is not None:
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            reshard_after_forward=False,
        )

    fully_shard(model, **fsdp_config)

    
    
def apply_compile(model: nn.Module | EaBNetQwen3):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # NOTE: This flag is needed for torch.compile to avoid graph breaking on dynamic shapes in token-choice MoE
    # but it is experimental.
    # torch._dynamo.config.capture_scalar_outputs = True
    for layer_id, transformer_block in model.layers.named_children():
        # TODO: remove when torch.compile supports fullgraph=True for MoE
        fullgraph = True
        transformer_block = torch.compile(transformer_block, fullgraph=fullgraph)
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")

