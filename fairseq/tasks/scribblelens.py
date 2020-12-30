# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import sys
import torch

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any
from omegaconf import MISSING

from fairseq.data import (AddTargetDataset, Dictionary, FileAudioDataset,
                          FileHandwritingDataset, HandwritingDictionary,
                          encoders)
from fairseq.data.data_utils import post_process
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig

from . import FairseqTask, LegacyFairseqTask, register_task
from .. import utils
from ..logging import metrics


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@dataclass
class ScribblelensConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: bool = field(
        default=False,
        metadata={"help": "if to return also labels from dataset"}
    )
    vocab_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to data directory"}
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    pad_to_multiples_of: Optional[int] = field(
        default=None,
        metadata={"help": "enforce that lengths of inputs are multiples of this"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to crop to for batching"}
    )



@register_task("scribblelens", dataclass=ScribblelensConfig)
class ScribblelensTask(FairseqTask):
    """"""

    cfg: ScribblelensConfig

    def __init__(
        self, 
        cfg: ScribblelensConfig, 
        source_dictionary=None,
        target_dictionary=None,
    ):
        super().__init__(cfg)
        self._target_dictionary = target_dictionary
        self._source_dictionary = source_dictionary
        # if cfg.eval_wer:
        #     assert cfg.labels is not None, "eval_wer can only be set during fine-tuning"
        self.blank_symbol = "*"

    @classmethod
    def setup_task(cls, cfg:ScribblelensConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (ScribblelensConfig): configuration of this task
        """
        if cfg.labels:
            target_dictionary = HandwritingDictionary(cfg.vocab_path)
        else:
            target_dictionary = None

        return cls(cfg, target_dictionary=target_dictionary)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg
        vocab_path = task_cfg.vocab_path if task_cfg.vocab_path is not None else task_cfg.data + '/tasman.alphabet.plus.space.mode5.json'

        self.datasets[split] = FileHandwritingDataset(
            task_cfg.data,
            vocab_path=vocab_path,
            split=split,
            max_sample_size=task_cfg.max_sample_size,
            min_sample_size=task_cfg.max_sample_size,
            pad_to_multiples_of=task_cfg.pad_to_multiples_of,
            min_length=task_cfg.min_sample_size,
            pad=task_cfg.labels is not None or task_cfg.enable_padding,
            normalize=task_cfg.normalize,
            labels=task_cfg.labels
        )


    @property
    def source_dictionary(self):
        return self._source_dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self._target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices
