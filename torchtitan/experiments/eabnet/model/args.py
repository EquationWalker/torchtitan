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
class EABNetArgs(BaseModelArgs):
    M: int = 8
    F: int = 161
    
    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        # TODO(jianiw): Add the number of flops for the autoencoder
        nparams = sum(p.numel() for p in model.parameters())
        logger.warning("EABNet model haven't implement get_nparams_and_flops() function")
        return nparams, 1
