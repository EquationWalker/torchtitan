# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Any
from torch.distributed.checkpoint.stateful import Stateful

from torch.utils.data import IterableDataset
from torchtitan.components.dataloader import ParallelAwareDataloader
import io
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger
from itertools import islice
import webdataset as wds
import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node



class MCSEDataset(IterableDataset, Stateful):
    """Dataset for FLUX text-to-image model.

    Args:
    dataset_name (str): Name of the dataset.
    dataset_path (str): Path to the dataset.
    model_transform (Transform): Callable that applies model-specific preprocessing to the sample.
    dp_rank (int): Data parallel rank.
    dp_world_size (int): Data parallel world size.
    infinite (bool): Whether to loop over the dataset infinitely.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        job_config: Optional[JobConfig] = None,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        shuffle_size: int = 5000
    ) -> None:

        def split_by_rank(src):
            yield from islice(src, dp_rank, None, dp_world_size)
 
        ds = wds.WebDataset(dataset_path, resampled=True, nodesplitter=None, workersplitter=split_by_rank)

        self.dataset_name = dataset_name
        self._data = ds.shuffle(shuffle_size).decode().to_tuple("noisy.pth", "clean.pth").unbatched()

    
        self.job_config = job_config

        self.infinite = infinite


    def _get_data_iter(self):
        return iter(self._data)

    def __iter__(self):
        dataset_iterator = self._get_data_iter()
        while True:
            # TODO: Add support for robust data loading and error handling.
            # Currently, we assume the dataset is well-formed and does not contain corrupted samples.
            # If a corrupted sample is encountered, the program will crash and throw an exception.
            # You can NOT try to catch the exception and continue, because the iterator within dataset
            # is not broken after raising an exception, so calling next() will throw StopIteration and might cause re-loop.
            try:
                sample = next(dataset_iterator)
            except StopIteration:
                # We are asumming the program hits here only when reaching the end of the dataset.
                if not self.infinite:
                    logger.warning(
                        f"Dataset {self.dataset_name} has run out of data. \
                         This might cause NCCL timeout if data parallelism is enabled."
                    )
                    break
                else:
                    # Reset offset for the next iteration if infinite
                    logger.warning(f"Dataset {self.dataset_name} is being re-looped.")
                    dataset_iterator = self._get_data_iter()
                    continue

            noisy, labels = sample

            yield dict(noisy=noisy), labels

    def load_state_dict(self, state_dict):
        pass

    def state_dict(self):
        return {}



# class MCSEDataset(IterableDataset, Stateful):
#     """Dataset for FLUX text-to-image model.

#     Args:
#     dataset_name (str): Name of the dataset.
#     dataset_path (str): Path to the dataset.
#     model_transform (Transform): Callable that applies model-specific preprocessing to the sample.
#     dp_rank (int): Data parallel rank.
#     dp_world_size (int): Data parallel world size.
#     infinite (bool): Whether to loop over the dataset infinitely.
#     """

#     def __init__(
#         self,
#         dataset_name: str,
#         dataset_path: Optional[str],
#         job_config: Optional[JobConfig] = None,
#         dp_rank: int = 0,
#         dp_world_size: int = 1,
#         infinite: bool = False,
#     ) -> None:

 
#         ds = load_dataset("webdataset", data_files={"train": dataset_path}, split="train", streaming=True)

#         self.dataset_name = dataset_name
#         self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

    
#         self.job_config = job_config

#         self.infinite = infinite

#         # Variables for checkpointing
#         self._sample_idx = 0
#         self._all_samples: list[dict[str, Any]] = []

#     def _get_data_iter(self):
#         if isinstance(self._data, Dataset):
#             if self._sample_idx == len(self._data):
#                 return iter([])
#             else:
#                 return iter(self._data.skip(self._sample_idx))

#         return iter(self._data)

#     def __iter__(self):
#         dataset_iterator = self._get_data_iter()
#         while True:
#             # TODO: Add support for robust data loading and error handling.
#             # Currently, we assume the dataset is well-formed and does not contain corrupted samples.
#             # If a corrupted sample is encountered, the program will crash and throw an exception.
#             # You can NOT try to catch the exception and continue, because the iterator within dataset
#             # is not broken after raising an exception, so calling next() will throw StopIteration and might cause re-loop.
#             try:
#                 sample = next(dataset_iterator)
#             except StopIteration:
#                 # We are asumming the program hits here only when reaching the end of the dataset.
#                 if not self.infinite:
#                     logger.warning(
#                         f"Dataset {self.dataset_name} has run out of data. \
#                          This might cause NCCL timeout if data parallelism is enabled."
#                     )
#                     break
#                 else:
#                     # Reset offset for the next iteration if infinite
#                     self._sample_idx = 0
#                     logger.warning(f"Dataset {self.dataset_name} is being re-looped.")
#                     dataset_iterator = self._get_data_iter()
#                     if not isinstance(self._data, Dataset):
#                         if hasattr(self._data, "set_epoch") and hasattr(
#                             self._data, "epoch"
#                         ):
#                             self._data.set_epoch(self._data.epoch + 1)
#                     continue

#             # print(f"Debug {type(sample)}-{type(sample["noisy.pth"][0])}")
#             sample_dict = {'noisy':torch.tensor(sample["noisy.pth"][0])}
#             # print(f"Debug-1 {sample_dict['noisy'].shape}*-{sample_dict['noisy'].dtype}")

#             self._sample_idx += 1

#             labels = torch.tensor(sample["clean.pth"][0])
#             # print(f"Debug labels {type(labels)} - {type(sample['noisy.pt'])}")

#             yield sample_dict, labels
            
#     def load_state_dict(self, state_dict):
#         if isinstance(self._data, Dataset):
#             self._sample_idx = state_dict["sample_idx"]
#         else:
#             assert "data" in state_dict
#             self._data.load_state_dict(state_dict["data"])

#     def state_dict(self):
#         if isinstance(self._data, Dataset):
#             return {"sample_idx": self._sample_idx}
#         else:
#             return {"data": self._data.state_dict()}




def build_mcse_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    # This parameter is not used, keep it for compatibility
    tokenizer: BaseTokenizer | None,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size


    ds = MCSEDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        job_config=job_config,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        # num_workers=1,
        pin_memory=True,
        # persistent_workers=True
    )

