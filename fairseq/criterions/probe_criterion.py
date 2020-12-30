# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging.meters import safe_round
from fairseq.models.probed_model import reduce_probe_metrics

@dataclass
class ProbeCriterionConfig(FairseqDataclass):
    pass


@register_criterion("probes", dataclass=ProbeCriterionConfig)
class ProbeCriterion(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        sample_size = 1

        probe_loss, probe_log_outs = model.get_probe_losses(sample)
        loss = probe_loss
        
        logging_output = {
            "loss": loss.item(),
            "ntokens": 1,
            "nsentences": 1,
            "sample_size": sample_size,
        }
        logging_output.update(probe_log_outs)

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        reduce_probe_metrics(logging_outputs, metrics)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
