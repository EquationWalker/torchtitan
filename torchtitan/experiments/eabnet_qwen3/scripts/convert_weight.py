import os
import torch
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import (
    _EmptyStateDictLoadPlanner,
)
from torch.distributed.checkpoint.metadata import (
    STATE_DICT_TYPE,
)
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.state_dict_saver import _save_state_dict


def dcp_to_torch_save(
    dcp_checkpoint_dir: str | os.PathLike,
    torch_save_path: str | os.PathLike,
    exclude_to_save: list[str] = [
        "optimizer",
        "lr_scheduler",
        "dataloader",
        "train_state",
    ],
):
    """
    Given a directory containing a DCP checkpoint, this function will convert it into a
    Torch save file.

    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint.
        torch_save_path: Filename to store the converted Torch save file.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """
    sd: STATE_DICT_TYPE = {}
    _load_state_dict(
        sd,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    sd = {k: v for k, v in sd.items() if k not in exclude_to_save}
    torch.save(sd, torch_save_path)


def torch_save_to_dcp(
    torch_save_path: str | os.PathLike,
    dcp_checkpoint_dir: str | os.PathLike,
):
    """
    Given the location of a torch save file, converts it into a DCP checkpoint.

    Args:
        torch_save_path: Filename of the Torch save file.
        dcp_checkpoint_dir: Directory to store the DCP checkpoint.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """

    state_dict = torch.load(torch_save_path, weights_only=False)
    # we don't need stateful behavior here because the expectation is anything loaded by
    # torch.load would not contain stateful objects.
    _save_state_dict(
        state_dict, storage_writer=FileSystemWriter(dcp_checkpoint_dir), no_dist=True
    )


# CHECKPOINT_DIR = "/data2/liuxin/expr/outputs/checkpoint/step-90000"
# TORCH_SAVE_CHECKPOINT_DIR = "/data2/liuxin/expr/outputs/torch_pth/90000.pth"

CHECKPOINT_DIR = "/data2/liuxin/expr/qwen_raw_loss_mask/checkpoint/step-590000"
TORCH_SAVE_CHECKPOINT_DIR = "/data2/liuxin/expr/qwen_raw_loss_mask/torch_pth/590000.pth"


dcp_to_torch_save(CHECKPOINT_DIR, TORCH_SAVE_CHECKPOINT_DIR)
