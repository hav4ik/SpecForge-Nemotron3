# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from datasets import Dataset
from specforge.distributed import get_draft_sp_group, get_sp_ulysses_group


class LengthBucketDistributedSampler(Sampler[int]):
    """
    Length-bucketed distributed sampler for DP training.

    Sorts the dataset by per-sample length, partitions the sorted index
    into contiguous "global batches" of size ``num_replicas * batch_size``,
    shuffles the *order* of those global batches per epoch (with a
    deterministic seed), and assigns each rank its slice of every global
    batch.

    Why bucket across ranks (not just within rank):
        With DP=N, each optimizer step is bottlenecked by the slowest
        rank. If rank 0 happens to draw a 32k-token sample while rank 1
        draws a 2k-token sample, rank 1 sits idle waiting for rank 0,
        and the step costs as much as the longest sample. Bucketing
        across ranks guarantees that all ranks within a single global
        step see samples of similar length, eliminating the straggler.

    Bias mitigation:
        Pure length-sorted iteration biases gradient updates by
        difficulty over an epoch (long samples first or last). To
        mitigate, the *order of global batches* is shuffled per epoch
        with a deterministic seed; the *intra-batch* order (which
        determines the per-step length) is intentionally NOT shuffled,
        so each step still sees bucketed lengths. This is the same
        tradeoff used by HF Trainer's ``group_by_length=True`` and
        fairseq's ``--required-batch-size-multiple`` length grouping.

        For training data with a hard floor on min length (e.g. >= 1024
        tokens), the worst-case length ratio within a global batch is
        bounded enough that the gradient bias is negligible compared to
        the throughput win. For data with extreme min/max ratios
        (e.g. min=10 tokens, max=65k tokens), bucketing is *not*
        recommended -- the per-step difficulty drift becomes too large.

    When NOT to use bucketing:
        * On-policy fine-tune (stage 2) where the data distribution is
          already narrow and you want maximum gradient diversity per
          step. Bucketing forces consecutive optimizer steps to see
          similar lengths, which is the opposite of what you want when
          you're trying to specialize precisely to that distribution.
        * Tiny datasets where dropping the trailing partial batch (which
          ``drop_last=True`` always does) loses meaningful samples.

    Args:
        lengths: Per-sample length array; ``len(lengths) == len(dataset)``.
                 Used only for sorting, not stored beyond ``__init__``.
        num_replicas: DP world size.
        rank: This process's DP rank.
        batch_size: Per-rank batch size (the ``DataLoader``'s
                    ``batch_size`` parameter, NOT global batch).
        seed: Base RNG seed; epoch is added to it for the per-epoch
              shuffle of global-batch order.
        drop_last: Always True in practice; the trailing partial global
                   batch is dropped to keep ranks balanced.
    """

    def __init__(
        self,
        lengths: List[int],
        num_replicas: int,
        rank: int,
        batch_size: int,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if not drop_last:
            # Supporting drop_last=False would require padding the final
            # global batch with re-sampled indices to keep ranks balanced;
            # we have no use case for it and it just adds bugs.
            raise ValueError("LengthBucketDistributedSampler requires drop_last=True")
        if num_replicas <= 0 or rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"invalid num_replicas={num_replicas} / rank={rank}"
            )
        if batch_size <= 0:
            raise ValueError(f"invalid batch_size={batch_size}")

        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        # Sort the dataset indices by length (stable on ties).
        self._sorted_indices = sorted(
            range(len(lengths)), key=lambda i: lengths[i]
        )
        # Drop the trailing partial global batch.
        global_batch = num_replicas * batch_size
        self._n_global_batches = len(self._sorted_indices) // global_batch
        self._sorted_indices = self._sorted_indices[
            : self._n_global_batches * global_batch
        ]
        # Per-rank epoch length (DataLoader will see this many indices).
        self._per_rank_len = self._n_global_batches * batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._per_rank_len

    def __iter__(self):
        gb = self.num_replicas * self.batch_size
        # Shuffle the ORDER of global batches deterministically per
        # epoch. Intra-batch order is preserved so length-bucketing
        # holds within each step.
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = torch.randperm(self._n_global_batches, generator=g).tolist()
        for batch_idx in order:
            start = batch_idx * gb
            rank_start = start + self.rank * self.batch_size
            rank_end = rank_start + self.batch_size
            for idx in self._sorted_indices[rank_start:rank_end]:
                yield idx


class DataCollatorWithPadding:
    """
    Datacollator that will dynamically pad the inputs for batching.
    """

    def __init__(self):
        self.sp_degree = torch.distributed.get_world_size(get_draft_sp_group())
        self.ulysses_degree = torch.distributed.get_world_size(get_sp_ulysses_group())

    def paddingtensor(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad to the longest sequence in the batch.

        Args:
            intensors: (B, n, S)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N, S)
        """
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(
            B, N - n, S, dtype=intensors.dtype, device=intensors.device
        )
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad 2D tensor to the longest sequence in the batch.

        Args:
            intensors: (B, n)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N)
        """
        B, n = intensors.shape
        padding_tensor = torch.zeros(
            B, N - n, dtype=intensors.dtype, device=intensors.device
        )
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of features.

        Args:
            features: A list of features, where each feature is a dictionary containing:
                - input_ids: torch.Tensor of shape (n,)
                - attention_mask: torch.Tensor of shape (n,)
                - loss_mask: torch.Tensor of shape (n,)

        Returns:
            A dictionary containing:
                - input_ids: torch.Tensor of shape (B, N)
                - attention_mask: torch.Tensor of shape (B, N)
                - loss_mask: torch.Tensor of shape (B, N)
        """
        max_length = max(item["input_ids"].shape[1] for item in features)

        # pad for sequence parrel
        max_length = (
            (max_length + self.sp_degree - 1) // self.sp_degree
        ) * self.sp_degree
        # position max len, ulysses do not need chuck position ids
        position_max_len = max_length * self.ulysses_degree

        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )
        if "position_ids" in features[0]:
            batch_position_ids = torch.cat(
                [
                    self.paddingtensor2D(item["position_ids"], position_max_len)
                    for item in features
                ]
            )
        else:
            batch_position_ids = None
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_state": None,
            "target": None,
        }
        if batch_position_ids is not None:
            batch["position_ids"] = batch_position_ids
        if all("hidden_state" in item for item in features):
            assert all(
                "target" in item for item in features
            ), "target is required when hidden_state is provided"
            if self.sp_degree > 1:  # USP mode
                batch["hidden_state"] = torch.cat(
                    [item["hidden_state"] for item in features]
                )
            else:
                batch["hidden_state"] = torch.cat(
                    [
                        self.paddingtensor(item["hidden_state"], max_length)
                        for item in features
                    ]
                )
            batch["target"] = torch.cat(
                [self.paddingtensor(item["target"], max_length) for item in features]
            )
        return batch


class VlmDataCollatorWithPadding:
    """
    Datacollator that will dynamically pad the inputs for batching.
    """

    def paddingtensor(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad to the longest sequence in the batch.

        Args:
            intensors: (B, n, S)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N, S)
        """
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad 2D tensor to the longest sequence in the batch.

        Args:
            intensors: (B, n)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N)
        """
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of features.

        Args:
            features: A list of features, where each feature is a dictionary containing:
                - input_ids: torch.Tensor of shape (n,)
                - attention_mask: torch.Tensor of shape (n,)
                - loss_mask: torch.Tensor of shape (n,)
                - pixel_values: torch.Tensor of shape (grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
                - image_grid_thw: torch.Tensor of shape (3,)

        Returns:
            A dictionary containing:
                - input_ids: torch.Tensor of shape (B, N)
                - attention_mask: torch.Tensor of shape (B, N)
                - loss_mask: torch.Tensor of shape (B, N)
        """
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )
        batch_pixel_values = torch.cat(
            [item["pixel_values"] for item in features], dim=0
        )
        batch_image_grid_thw = torch.cat(
            [item["image_grid_thw"] for item in features], dim=0
        )
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "pixel_values": batch_pixel_values,
            "image_grid_thw": batch_image_grid_thw,
            "hidden_state": None,
            "target": None,
        }
        if all("hidden_state" in item for item in features):
            assert all(
                "target" in item for item in features
            ), "target is required when hidden_state is provided"
            batch["hidden_state"] = torch.cat(
                [
                    self.paddingtensor(item["hidden_state"], max_length)
                    for item in features
                ]
            )
            batch["target"] = torch.cat(
                [self.paddingtensor(item["target"], max_length) for item in features]
            )
        return batch


def prepare_dp_dataloaders(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    process_group: Optional[dist.ProcessGroup] = None,
    pin_memory: Optional[bool] = False,
    shuffle: Optional[bool] = False,
    is_vlm: Optional[bool] = False,
    prefetch_factor: Optional[int] = 2,
    length_bucket_sample_lengths: Optional[List[int]] = None,
    length_bucket_seed: int = 42,
    **dataloader_kwargs,
) -> DataLoader:
    """
    Prepare dataloader for distributed data parallel training.

    Args:
        dataset: The dataset to load data from.
        batch_size: The batch size for each GPU.
        num_workers: The number of workers for data loading.
        process_group: The process group for distributed training.
        pin_memory: Whether to pin memory for data loading.
        shuffle: Whether to shuffle the dataset.
        is_vlm: Whether the dataset is a vision-language model dataset.
        **dataloader_kwargs: Additional keyword arguments for the DataLoader.

    Returns:
        A DataLoader for the dataset.
    """
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    if length_bucket_sample_lengths is not None:
        if len(length_bucket_sample_lengths) != len(dataset):
            raise ValueError(
                f"length_bucket_sample_lengths has "
                f"{len(length_bucket_sample_lengths)} entries but dataset "
                f"has {len(dataset)} samples"
            )
        sampler = LengthBucketDistributedSampler(
            lengths=length_bucket_sample_lengths,
            num_replicas=world_size,
            rank=rank,
            batch_size=batch_size,
            seed=length_bucket_seed,
            drop_last=True,
        )
    else:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
    if is_vlm:
        datacollator_cls = VlmDataCollatorWithPadding
    else:
        datacollator_cls = DataCollatorWithPadding

    if num_workers == 0:
        prefetch_factor = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=datacollator_cls(),
        drop_last=True,
        **dataloader_kwargs,
    )
    return dataloader


def parse_harmony_message_content(content):
    """
    解析 content 字符串中的 Harmony 格式。
    如果匹配到 Harmony 格式，返回包含 channel 和 content 的列表；
    否则，返回原内容并标记为默认 channel。
    """
    # 匹配 <|channel|>xxx<|message|>yyy<|end|>
    pattern = r"<\|channel\|>(.*?)<\|message\|>(.*?)<\|end|>"
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        # 如果没有匹配到 Harmony 标签，视作普通文本
        return [{"channel": "text", "content": content}]

    results = []
    for channel, msg_body in matches:
        results.append({"channel": channel.strip(), "content": msg_body.strip()})
    return results


def process_harmony_conversations(conversation):
    """
    处理传入的 list[list[dict]] 结构
    """
    new_conversation = []
    for msg in conversation:
        role = msg.get("role")
        original_content = msg.get("content", "")

        # 解析 content 中的 Harmony 结构
        segments = parse_harmony_message_content(original_content)

        # 为每个解析出的通道生成一个新的消息字典
        for seg in segments:
            new_msg = {
                "role": role,
                "channel": seg["channel"],  # 新增字段标识通道
                "content": seg["content"],
            }
            new_conversation.append(new_msg)

    return new_conversation
