# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional, Iterable

import torch
from .infra.utils import no_grad_sync
from torchtitan.config import ConfigManager, JobConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer



class EABNetTrainer(Trainer):
    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        # For Flux model, we need distinct seed across FSDP ranks to ensure we randomly dropout prompts info in dataloader
        dist_utils.set_determinism(
            self.parallel_dims.world_mesh,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
            distinct_seed_mesh_dim="dp_shard",
        )

        # NOTE: self._dtype is the data type used for encoders (image encoder, T5 text encoder, CLIP text encoder).
        # We cast the encoders and it's input/output to this dtype.  If FSDP with mixed precision training is not used,
        # the dtype for encoders is torch.float32 (default dtype for Flux Model).
        # Otherwise, we use the same dtype as mixed precision training process.
        # self._dtype = (
        #     TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
        #     if self.parallel_dims.dp_shard_enabled
        #     else torch.float32
        # )
        
        # self.stft_params = {'sr': 16000,
        #         'win_size_sec': 0.02,
        #         'win_shift_sec': 0.01,
        #         'num_fft': 320}

    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        # explicitly convert flux model to be Bfloat16 no matter FSDP is applied or not
        model = self.model_parts[0]

        # Mixed precision training is handled by fully_shard
        # with self.maybe_enable_amp:
        esti_stft = model(input_dict['noisy'])
        
        loss = self.loss_fn(esti_stft, labels)
        # pred.shape=(bs, seq_len, vocab_size)
        # need to free to before bwd to avoid peaking memory
        del esti_stft
        loss.backward()

        return loss
    



if __name__ == "__main__":
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Optional[EABNetTrainer] = None

    try:
        trainer = EABNetTrainer(config)
        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        torch.distributed.destroy_process_group()
        logger.info("Process group destroyed.")
