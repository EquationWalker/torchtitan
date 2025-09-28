# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torch import nn
from torchtitan.protocols import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class Qwen3ModelArgs(BaseModelArgs):
    M: int = 8
    F: int = 161
    
    dim: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    hidden_dim: int = 3072

    norm_eps: float = 1e-6
    rope_theta: float = 1000000
    qk_norm: bool = True
    max_seq_len: int = 4096
    depth_init: bool = True

    use_flex_attn: bool = False
    attn_mask_type: str = "causal"

    enable_weight_tying: bool = False
    
    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        # TODO(jianiw): Add the number of flops for the autoencoder
        nparams = sum(p.numel() for p in model.parameters())
        logger.warning("EABNet model haven't implement get_nparams_and_flops() function")
        return nparams, 1
