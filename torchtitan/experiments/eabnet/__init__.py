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
from .model.args import EABNetArgs
from .model.model import EaBNet

# __all__ = [
#     "FluxModelArgs",
#     "FluxModel",
#     "flux_configs",
#     "parallelize_flux",
# ]


flux_configs = {
    "base": EABNetArgs()
}


register_train_spec(
    TrainSpec(
        name="EABNet",
        model_cls=EaBNet,
        model_args=flux_configs,
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
