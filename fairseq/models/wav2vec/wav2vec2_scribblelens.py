# Copyright (c) Facebook, Inc. and its affiliates., UWr
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model, probed_model
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange

from .wav2vec2 import Wav2Vec2Config


@dataclass
class Wav2Vec2SLConfig(Wav2Vec2Config):
    probe_defs: Optional[Dict[str, Any]] = field(default=None, metadata={"help": "probes"})
    num_channels: int = field(default=1, metadata={"help": "number of input channels"})
    compute_alignment_metrics: bool = field(default=False, metadata={"help": "compute mutual info and rand scores"})

@register_model("wav2vec2_scribblelens", dataclass=Wav2Vec2SLConfig)
class Wav2Vec2ModelSL(BaseFairseqModel, probed_model.ProbedModel):
    def __init__(self, cfg: Wav2Vec2Config):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            in_d=cfg.num_channels,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and not cfg.quantize_input
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.logit_temp = cfg.logit_temp

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        if cfg.quantize_targets:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        if cfg.quantize_input:
            if cfg.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else cfg.encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfg.latent_vars,
                    temp=cfg.latent_temp,
                    groups=cfg.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
            self.project_inp = nn.Linear(vq_dim, cfg.encoder_embed_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        self.attach_probes(cfg.probe_defs)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2SLConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def sample_negatives(self, y, num):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        cross_high = tsz * bsz
        high = tsz
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits /= self.logit_temp

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        return logits

    def forward(self, source, padding_mask=None, mask=True, features_only=False, alignments=None):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        compute_alignment_metrics = self.cfg.compute_alignment_metrics and alignments is not None

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            assert padding_mask.size(1) == 1
            padding_mask = padding_mask.squeeze(1)
            scale = padding_mask.size(1) // features.size(1)
            extra = padding_mask.size(1) % features.size(1) # should be 0 since 1st CNN reduces number of features [scale] times (due to the architecture choice)
            assert extra == 0
            padding_mask = padding_mask[:, ::scale]
            assert np.all(padding_mask.shape == features.shape[:-1])
            if compute_alignment_metrics:
                alignments = alignments[:, ::scale]

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        # [!] careful with some nasty indirect dependencies - changing this function arg padding_mask -> sth else
        #     seems to break the code completely because indirect dependencies on passthrough **kwargs etc. or so are so horrible

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
                if compute_alignment_metrics:
                    alignments = alignments[mask_indices]
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x = self.encoder(x, padding_mask=padding_mask)
        
        if features_only:
            return {"x": x, "padding_mask": padding_mask}

        if self.quantizer:
            q = self.quantizer(y, produce_targets=compute_alignment_metrics)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands, *_ = self.quantizer(unmasked_features, produce_targets=False)
                negs, _ = self.sample_negatives(neg_cands, y.size(1))
                negs = self.project_q(negs)

            else:
                negs, _ = self.sample_negatives(y, y.size(1))

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(unmasked_features, y.size(1))
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(y, y.size(1))

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {"x": x, "padding_mask": padding_mask, "features_pen": features_pen}

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        if compute_alignment_metrics:
            result = {
                **result,
                **self.get_alignment_metrics(q["targets"], alignments)
            }

        return result

    def get_alignment_metrics(self, targets, ali_gt):
        # We currently only quantize unmasked features
        import sklearn.metrics

        with torch.no_grad():
            targets = targets.reshape(-1, self.quantizer.groups)
            targetts = targets.detach().cpu().numpy()
            ali_es = targets[:, 0] + self.quantizer.num_vars * targets[:, 1]
            ali_es = ali_es.detach().cpu().numpy()
            ali_gt = ali_gt.detach().cpu().numpy()

            return {
                "adjusted_mutual_info": 
                    sklearn.metrics.adjusted_mutual_info_score(ali_gt, ali_es, average_method='arithmetic'),
                "normalized_mutual_info": 
                    sklearn.metrics.normalized_mutual_info_score(ali_gt, ali_es, average_method='arithmetic'),
                "adjusted_rand_score":
                    sklearn.metrics.adjusted_rand_score(ali_gt, ali_es)
            }


    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, source, padding_mask, mask=False):
        res = self.forward(source, padding_mask, mask=mask, features_only=True)
        return res["x"], res["padding_mask"]

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List,
        in_d: int=1,
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            padding,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                assert len(k) == 2
                conv = nn.Conv2d(n_in, n_out, k, stride=stride, bias=conv_bias, padding=padding)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                assert False  # JCh: didn't check teh shapes
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 4, "invalid conv definition: " + str(cl)
            (dim, k, stride, padding) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    padding,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # print(x.shape)
        # import sys; sys.exit(1)
        if len(x.shape) == 3:
            # BxHxW -> BxCxHxW
            x = x.unsqueeze(1)
        
        # BxCxHxW -> BxCxWxH
        x = x.transpose(-2, -1).contiguous()

        for conv in self.conv_layers:
            x = conv(x)
        
        assert x.shape[-1] == 1
        x = x.squeeze(3)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )
                for _ in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None):
        x = self.extract_features(x, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x, padding_mask=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                layer_results.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn
