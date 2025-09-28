# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .dataset.mcse_dataset import build_mcse_dataloader
from .infra.parallelize import parallelize_eabnet
from .loss import build_mse_loss
from .model.args import Qwen3ModelArgs
from .model.model import EaBNetQwen3

__all__ = [
    "Qwen3ModelArgs",
    "EaBNetQwen3",
    "eabnet_qwen3_configs",
    "parallelize_eabnet",
]


eabnet_qwen3_configs = {
    "0.6B": Qwen3ModelArgs(
        max_seq_len=2402,
        head_dim=128,
        dim=1024,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=3072,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
     "1.4B": Qwen3ModelArgs(

        max_seq_len=2402,
        head_dim=128,
        dim=2048,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=6144,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "3B": Qwen3ModelArgs(
        max_seq_len=2402,
        head_dim=128,
        dim=2400,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=9600,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
}


register_train_spec(
    TrainSpec(
        name="EABNetQwen3",
        model_cls=EaBNetQwen3,
        model_args=eabnet_qwen3_configs,
        parallelize_fn=parallelize_eabnet,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_mcse_dataloader,
        build_tokenizer_fn=None,
        build_loss_fn=build_mse_loss,
        build_validator_fn=None,
        state_dict_adapter=None,
    )
)
