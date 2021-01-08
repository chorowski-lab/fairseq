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
from typing import Optional, Any, Dict

from fairseq.data.audio.speech_features_dataset import SpeechFeaturesDataset

from .audio_pretraining import (
    AudioPretrainingTask, AudioPretrainingConfig,
    LabelEncoder, FairseqDataclass, register_task, AddTargetDataset)


@dataclass
class AudioPretrainingFeatsConfig(AudioPretrainingConfig):
    pad_to_multiples_of: Optional[int] = field(
        default=None,
        metadata={"help": "enforce that lengths of inputs are multiples of this"}
    )
    transform: str = field(
        default='fbank',
        metadata={"help": "kind of features to use"}
    )
    transform_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "args for the featutres extractor"}
    )


@register_task("audio_pretraining_feats", dataclass=AudioPretrainingFeatsConfig)
class AudioPretrainingFeaturesTask(AudioPretrainingTask):
    """"""

    cfg: AudioPretrainingConfig

    def __init__(
        self,
        cfg: AudioPretrainingConfig,
        source_dictionary=None,
        target_dictionary=None,
    ):
        super().__init__(cfg, source_dictionary=source_dictionary, target_dictionary=target_dictionary)
        
    # @classmethod
    # def setup_task(cls, cfg: AudioPretrainingConfig, **kwargs):
    #     """Setup the task (e.g., load dictionaries).

    #     Args:
    #         cfg (AudioPretrainingConfig): configuration of this task
    #     """

    #     if cfg.labels:
    #         dict_path = os.path.join(cfg.data, f"dict.{cfg.labels}.txt")
    #         target_dictionary = Dictionary.load(dict_path)
    #     else:
    #         target_dictionary = None

    #     return cls(cfg, target_dictionary=target_dictionary)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == 'ctc'

        manifest = os.path.join(data_path, "{}.tsv".format(split))
        self.datasets[split] = SpeechFeaturesDataset(
            manifest,
            transform=task_cfg.transform,
            transform_kwargs=task_cfg.transform_kwargs,
            sample_rate=task_cfg.sample_rate,
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.max_sample_size,
            min_length=self.cfg.min_sample_size,
            pad=task_cfg.labels is not None or task_cfg.enable_padding,
            pad_to_multiples_of=task_cfg.pad_to_multiples_of,
            normalize=task_cfg.normalize,
        )

        if task_cfg.labels:
            label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
            labels = []
            with open(label_path, "r") as f:
                for line in f:
                    labels.append(line)

            process_label = LabelEncoder(self.target_dictionary)

            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                add_to_input=task_cfg.autoregressive,
            )

