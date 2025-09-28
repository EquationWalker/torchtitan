# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import Any, Optional, TypeVar, Union, Iterable

import torch
import torch.nn as nn

from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer

from torchtitan.components.optimizer import OptimizerConfig, OptimizersContainer
from torchtitan.components.ft import FTManager, has_torchft
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims

__all__ = [
    "GSOptimizersContainer",
    "build_optimizers"
]


# if has_torchft:
#     import torchft as ft


T = TypeVar("T", bound=Optimizer)


class GradScalerOptimizer(torch.optim.Optimizer):
    """
    Internal wrapper around a torch optimizer.

    Args:
        optimizer (`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
        scaler (`torch.amp.GradScaler` or `torch.cuda.amp.GradScaler`, *optional*):
            The scaler to use in the step function if training with mixed precision.
    """

    def __init__(self, optimizer, scaler=None):
        self.optimizer = optimizer
        self.scaler = scaler
        self._is_overflow = False

        if self.scaler is not None:
            self._optim_step_called = False
            self._optimizer_original_step_method = self.optimizer.step
            self._optimizer_patched_step_method = patch_optimizer_step(self, self.optimizer.step)

    @property
    def state(self):
        return self.optimizer.state

    @state.setter
    def state(self, state):
        self.optimizer.state = state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups):
        self.optimizer.param_groups = param_groups

    @property
    def defaults(self):
        return self.optimizer.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optimizer.defaults = defaults

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self, set_to_none=None):
        accept_arg = "set_to_none" in inspect.signature(self.optimizer.zero_grad).parameters
        if accept_arg:
            if set_to_none is None:
                set_to_none = True
            self.optimizer.zero_grad(set_to_none=set_to_none)
        else:
            if set_to_none is not None:
                raise ValueError("`set_to_none` for Optimizer.zero_grad` is not supported by this optimizer.")
            self.optimizer.zero_grad()

    def train(self):
        """
        Sets the optimizer to "train" mode. Useful for optimizers like `schedule_free`
        """
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

    def eval(self):
        """
        Sets the optimizer to "eval" mode. Useful for optimizers like `schedule_free`
        """
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

    def step(self, *args, **kwargs):
        if self.scaler is not None:
            self.optimizer.step = self._optimizer_patched_step_method

            self.scaler.step(self.optimizer, *args, **kwargs)
            self.scaler.update()

            if not self._optim_step_called:
                # If the optimizer step was skipped, gradient overflow was detected.
                self._is_overflow = True
            else:
                self._is_overflow = False
            # Reset the step method to the original one
            self.optimizer.step = self._optimizer_original_step_method
            # Reset the indicator
            self._optim_step_called = False
        else:
            self.optimizer.step(*args, **kwargs)


    def _switch_parameters(self, parameters_map):
        for param_group in self.optimizer.param_groups:
            param_group["params"] = [parameters_map.get(p, p) for p in param_group["params"]]

    @property
    def step_was_skipped(self):
        """Whether or not the optimizer step was skipped."""
        return self._is_overflow

    def __getstate__(self):
        _ignored_keys = [
            "_optim_step_called",
            "_optimizer_original_step_method",
            "_optimizer_patched_step_method",
        ]
        return {k: v for k, v in self.__dict__.items() if k not in _ignored_keys}

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.scaler is not None:
            self._optim_step_called = False
            self._optimizer_original_step_method = self.optimizer.step
            self._optimizer_patched_step_method = patch_optimizer_step(self, self.optimizer.step)


def patch_optimizer_step(grad_scaler_optimizer: GradScalerOptimizer, method):
    def patched_step(*args, **kwargs):
        grad_scaler_optimizer._optim_step_called = True
        return method(*args, **kwargs)

    return patched_step


class GSOptimizersContainer(OptimizersContainer):

    scaler: Optional[GradScaler]

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        scaler_kwargs: dict[str, Any] = dict(enabled=True)
    ) -> None:
        self.scaler = GradScaler(**scaler_kwargs) if scaler_kwargs['enabled'] else None
        
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        for model in self.model_parts:
            params = [p for p in model.parameters() if p.requires_grad]
            self.optimizers.append(GradScalerOptimizer(optimizer_cls(params, **optimizer_kwargs),
                                                       scaler=self.scaler))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)

    def state_dict(self) -> dict[str, Any]:
        sd = {'others': super().state_dict()}
        if self.scaler:
            sd['scaler'] = self.scaler.state_dict()
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict['others'])
        if self.scaler:
            assert 'scaler' in state_dict, 'ckpt not have `scaler` pth.'
            self.scaler.load_state_dict(state_dict['scaler'])
         
    def unscale_gradients(self):
        if not self.scaler:
            return
        for opt in self.optimizers:
            self.scaler.unscale_(opt.optimizer)
                
    def scale(
        self,
        outputs: Union[torch.Tensor, Iterable[torch.Tensor]],
    ) -> Union[torch.Tensor, Iterable[torch.Tensor]]:
        return self.scaler.scale(outputs) if self.scaler else outputs


def build_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``optimizer_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer`` and
    ``OptimizersInBackwardContainer``.

    **Note**
    Users who want to customize the optimizer behavior can create their own
    ``OptimizersContainer`` subclass and ``build_optimizers``. Passing the
    customized ``build_optimizers`` to ``TrainSpec`` will create the customized
    ``OptimizersContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_config (OptimizerConfig): Optimizer config containing the optimizer name and parameters.
        parallel_dims (ParallelDims): Parallel dimensions for the model.
    """
    optim_in_bwd = optimizer_config.early_step_in_backward
    if optim_in_bwd:
        if parallel_dims.ep_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Expert Parallel."
            )
        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Pipeline Parallel."
            )
        if ft_manager and ft_manager.enabled:
            raise NotImplementedError(
                "TorchFT is not supported with optimizers in backward."
            )

    name = optimizer_config.name
    lr = optimizer_config.lr
    beta1 = optimizer_config.beta1
    beta2 = optimizer_config.beta2
    eps = optimizer_config.eps
    weight_decay = optimizer_config.weight_decay

    optim_implementation = optimizer_config.implementation
    assert optim_implementation in ["fused", "foreach", "for-loop"]

    fused = optim_implementation == "fused"
    foreach = optim_implementation == "foreach"

    optimizer_kwargs = {
        "lr": lr,
        "betas": (beta1, beta2),
        "eps": eps,
        "weight_decay": weight_decay,
        "fused": fused,
        "foreach": foreach,
    }

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")
    optimizer_cls = optimizer_classes[name]

    if optim_in_bwd:
       pass

    if ft_manager and ft_manager.enabled:
        pass

    return GSOptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)

