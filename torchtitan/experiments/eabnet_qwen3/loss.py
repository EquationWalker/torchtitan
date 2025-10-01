# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


def stft_mag_loss(esti, label):
    """
    esti : b, f, t, 2
    label: b, f, t, 2
    """
    mag_esti, mag_label = torch.norm(esti, dim=-1), torch.norm(label, dim=-1)
    loss = F.mse_loss(mag_esti, mag_label)
    return loss


def stft_loss(esti, label):
    """
    esti : b, f, t, 2
    label: b, f, t, 2
    """
    return 0.5 * F.mse_loss(esti, label) + 0.5 * stft_mag_loss(esti, label)


def build_mse_loss(job_config: JobConfig):
    loss_fn = stft_loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn)
    return loss_fn
