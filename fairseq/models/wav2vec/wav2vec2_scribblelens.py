# Copyright (c) Facebook, Inc. and its affiliates., UWr
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import BaseFairseqModel, register_model, register_model_architecture
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
    HierarchicalSegmentationLayer,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange

import random
from PIL import Image, ImageDraw

@register_model("wav2vec2_scribblelens")
class Wav2Vec2ModelSL(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

        parser.add_argument(
            "--extractor-mode",
            choices=["default", "layer_norm"],
            help="mode for feature extractor. default has a single group norm with d groups in the first conv block, whereas layer_norm has layer norms in every block (meant to use with --normalize)",
        )

        parser.add_argument(
            "--encoder-layers",
            type=int,
            metavar="L",
            help="num encoder layers in the transformer",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )

        parser.add_argument(
            "--dropout",
            type=float,
            metavar="D",
            help="dropout probability for the transformer",
        )

        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )

        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )

        parser.add_argument(
            "--final-dim",
            type=int,
            metavar="D",
            help="project final representations and targets to this many dimensions",
        )

        parser.add_argument(
            "--layer-norm-first",
            action="store_true",
            help="apply layernorm first in the transformer",
        )

        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            help="probability of dropping a tarnsformer layer",
        )

        parser.add_argument(
            "--conv-feature-layers",
            type=str,
            metavar="EXPR",
            help="convolutional feature extraction layers [(dim, kernel_size, stride), ...]",
        )

        parser.add_argument(
            "--logit-temp", type=float, help="temperature to divide logits by"
        )

        parser.add_argument(
            "--quantize-targets", action="store_true", help="use quantized targets"
        )

        parser.add_argument(
            "--quantize-input", action="store_true", help="use quantized inputs"
        )

        parser.add_argument(
            "--same-quantizer",
            action="store_true",
            help="use same quantizer for inputs and targets",
        )

        parser.add_argument(
            "--feature-grad-mult",
            type=float,
            help="multiply feature extractor var grads by this",
        )

        parser.add_argument(
            "--latent-vars",
            type=int,
            metavar="N",
            help="number of latent variables V in each group of the codebook",
        )

        parser.add_argument(
            "--latent-groups",
            type=int,
            metavar="N",
            help="number of groups G of latent variables in the codebook",
        )

        parser.add_argument(
            "--latent-dim",
            type=int,
            metavar="N",
            help="if set, uses this dimensionality for latent variables. otherwise uses final_dim / latent_groups",
        )

        parser.add_argument("--mask-length", type=int, help="mask length")

        parser.add_argument(
            "--mask-prob", type=float, help="probability of replacing a token with mask"
        )

        parser.add_argument(
            "--mask-selection",
            type=str,
            choices=["static", "uniform", "normal", "poisson"],
            help="how to choose masks",
        )

        parser.add_argument(
            "--mask-other",
            type=float,
            help="secondary mask argument (used for more complex distributions), see help in compute_mask_indices",
        )

        parser.add_argument(
            "--no-mask-overlap",
            action="store_true",
            help="whether to allow masks to overlap",
        )

        parser.add_argument(
            "--mask-min-space",
            type=int,
            help="min space between spans (if no overlap is enabled)",
        )

        parser.add_argument(
            "--mask-channel-length",
            type=int,
            help="repeat the mask indices multiple times",
        )

        parser.add_argument(
            "--mask-channel-prob",
            type=float,
            help="probability of replacing a token with mask",
        )

        parser.add_argument(
            "--mask-channel-selection",
            type=str,
            choices=["static", "uniform", "normal", "poisson"],
            help="how to choose masks",
        )

        parser.add_argument(
            "--mask-channel-other",
            type=float,
            help="secondary mask argument (used for more complex distributions), see help in compute_mask_indices",
        )

        parser.add_argument(
            "--no-mask-channel-overlap",
            action="store_true",
            help="whether to allow masks to overlap",
        )

        parser.add_argument(
            "--mask-channel-min-space",
            type=int,
            help="min space between spans (if no overlap is enabled)",
        )

        parser.add_argument(
            "--dropout-input",
            type=float,
            metavar="D",
            help="dropout to apply to the input (after feat extr)",
        )

        parser.add_argument(
            "--dropout-features",
            type=float,
            metavar="D",
            help="dropout to apply to the features (after feat extr)",
        )

        parser.add_argument(
            "--num-negatives", type=int, metavar="N", help="number of negative examples"
        )

        parser.add_argument(
            "--negatives-from-everywhere",
            action="store_true",
            help="sample negatives from everywhere, not just masked states",
        )

        parser.add_argument(
            "--cross-sample-negatives",
            type=int,
            metavar="N",
            help="num of cross sampled negatives",
        )

        parser.add_argument(
            "--codebook-negatives",
            type=int,
            metavar="N",
            help="num of codebook sampled negatives",
        )

        parser.add_argument(
            "--conv-pos",
            type=int,
            metavar="N",
            help="number of filters for convolutional positional embeddings",
        )

        parser.add_argument(
            "--conv-pos-groups",
            type=int,
            metavar="N",
            help="number of groups for convolutional positional embedding",
        )

        parser.add_argument(
            "--latent-temp",
            type=str,
            metavar="D",
            help="temperature for latent variable sampling. can be tuple of 3 values (start, end, decay)",
        )

        parser.add_argument(
            "--target-glu", action="store_true", help="adds projection + glu to targets"
        )

        parser.add_argument(
            "--conv-bias", action="store_true", help="include bias in conv encoder"
        )

        parser.add_argument(
            "--segm", type=str, help="use segmentation on representations; 'hier' (without ') for hierarchical segm; " \
                + "also contains options, e.g. for var format is hier:<segment_cost>:<rounding_loss>:<shortening_mode>:<batchavg_segment_nr_per_line>, where:\n" \
                + " i) <segment_cost> is se (squared error), var (variance, se div by length), cos (cosine similarity mapped linearly to distance metric and scaled with segment length) \n" \
                + " ii) <rounding_loss> is additional rounding loss to use (se, var, lin, cos, or none) - measuring distance of average given for segment from original representations; " \
                + "need to add weight for this loss in loss-weights param if not none\n" \
                + " iii) <shortening_mode> is one of: shorten (averages in segments and replaces each with length 1), orig_len (replace with mean in segments, but keep length), " \
                + "orig_len+guess_orig (as in orig_len, but use original not-averaged representations as masked ones to guess correct one from)\n" \
                + " iv) <batchavg_segment_nr_reduction_per_line> is/are float/floats of format <avg_reduction> or <min_avg_reduction>-<max_avg_reduction>\n"
        )  # TODO maybe also think about an option with ~constant length reduction (but at least one piece each segment) so that 
           #      the long segments are not complete random averaged stuff

        parser.add_argument(
            "--log-ids", type=str, help="for what ids to log, format: <operator>:arg1,<operator>:arg1:arg2,... without spaces, " \
            + "operator can be =(id) [=:id] or %(X, id) [%:1000:0], meaning exact id or ids of that modulo X"
        )

        parser.add_argument(
            "--random-log-freq", type=float, help="how frequently (pbb) to log for randomly chosen ids"
        )

        # to have some data logged, need to specify for which IDs (--log-ids and/or --random-log-freq) and what to log (flags below)

        parser.add_argument(
            "--segm-log-dir", type=str, help="where to log chosen segmentation images; also serves as 'do log' flag"
        )

        parser.add_argument(
            "--repr-data-log-dir", type=str, help="where to log chosen array data (representation data, raw input images, and segment borders if segmentation); also serves as 'do log' flag"
        )

        

    def __init__(self, args):
        super().__init__()
        self.args = args

        feature_enc_layers = eval(args.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=args.extractor_mode,
            conv_bias=args.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, args.encoder_embed_dim)
            if self.embed != args.encoder_embed_dim and not args.quantize_input
            else None
        )

        self.mask_prob = args.mask_prob
        self.mask_selection = args.mask_selection
        self.mask_other = args.mask_other
        self.mask_length = args.mask_length
        self.no_mask_overlap = args.no_mask_overlap
        self.mask_min_space = args.mask_min_space

        self.mask_channel_prob = args.mask_channel_prob
        self.mask_channel_selection = args.mask_channel_selection
        self.mask_channel_other = args.mask_channel_other
        self.mask_channel_length = args.mask_channel_length
        self.no_mask_channel_overlap = args.no_mask_channel_overlap
        self.mask_channel_min_space = args.mask_channel_min_space

        self.dropout_input = nn.Dropout(args.dropout_input)
        self.dropout_features = nn.Dropout(args.dropout_features)

        self.feature_grad_mult = args.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = args.num_negatives
        self.cross_sample_negatives = args.cross_sample_negatives
        self.codebook_negatives = args.codebook_negatives
        self.negatives_from_everywhere = args.negatives_from_everywhere

        self.logit_temp = args.logit_temp

        final_dim = args.final_dim if args.final_dim > 0 else args.encoder_embed_dim

        if args.quantize_targets:
            vq_dim = args.latent_dim if args.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=args.latent_vars,
                temp=eval(args.latent_temp),
                groups=args.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        if args.quantize_input:
            if args.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = (
                    args.latent_dim if args.latent_dim > 0 else args.encoder_embed_dim
                )
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=args.latent_vars,
                    temp=eval(args.latent_temp),
                    groups=args.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
            self.project_inp = nn.Linear(vq_dim, args.encoder_embed_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(args.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(args)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if args.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(args.encoder_embed_dim, final_dim)

        # options for choosing ids to log for
        self.random_log_freq = args.random_log_freq if 'random_log_freq' in args else None
        if 'log_ids' in args:
            options = args.log_ids.split(",")
            self.log_ids = []
            for opt in options:
                details = opt.split(':')
                if details[0] == "%":
                    # need to bind details like that, because otherwise details variable will be bound to some random thing
                    # that was used later and was also named details; 
                    # imo one of the nastiest things in python, 
                    # as scope in python is "until end of function" and not "until the end of the function or sth"
                    self.log_ids.append((lambda details: (lambda x: x % int(details[1]) == int(details[2])))(details))
                elif details[0] == '=':
                    self.log_ids.append((lambda details: (lambda x: x == int(details[1])))(details))
                else:
                    assert False
        else:
            self.log_ids = None

        # part for supported segmentation options
        if 'segm' in args:
            segm_opts = args.segm.split(":")
            # this part needs to set stuff needed by 'segmentation' method
            if segm_opts[0] == "hier":
                self.segm = "var"
                assert len(segm_opts) == 5
                self.hier_segm_merge_priority = segm_opts[1]
                self.hier_segm_rounding_loss = segm_opts[2] if segm_opts[2] != "none" else None
                shorten_opts = segm_opts[3].split("+")
                self.hier_segm_shortening_policy = shorten_opts[0]
                self.hier_segm_guess_orig = len(shorten_opts) > 1 and shorten_opts[1] == "guess_orig"
                length_reduction_options = list(map(float, segm_opts[4].split("-")))
                if len(length_reduction_options) == 1:
                    self.hier_segm_strict_reduction = length_reduction_options[0]
                    self.hier_segm_reduction_range = None
                elif len(length_reduction_options) == 2:
                    self.hier_segm_strict_reduction = None
                    assert length_reduction_options[0] <= length_reduction_options[1]
                    self.hier_segm_reduction_range = tuple(length_reduction_options)
                else:
                    assert False
            else:
                assert False  # for now only that supported
            if 'segm_log_dir' in args:
                self.segm_log_dir = args.segm_log_dir
            else:
                self.segm_log_dir = None
        else:
            self.segm = None
            self.segm_log_dir = None

        if 'repr_data_log_dir' in args:
            self.repr_data_log_dir = args.repr_data_log_dir
        else:
            self.repr_data_log_dir = None

        self.need_logging = self.segm_log_dir is not None or self.repr_data_log_dir is not None

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, args, task=None):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        return cls(args)

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

    def forward(self, source, padding_mask=None, mask=True, features_only=False, id=None, epoch=None):
        # padding_mask = None  # JCh: padding_mask prob need to be True where the data is padded. mask=True => data invalid

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        # features = torch.squeeze(features)  # TODO check if this makes sense; also seems length reduction is too big for this input
        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        # unmasked_features = features.clone() needed to move after segmentation

        if padding_mask is not None:
            assert padding_mask.size(1) == 1
            padding_mask = padding_mask.squeeze(1)
            scale_float = float(padding_mask.size(1)) / features.size(1) 
            scale = padding_mask.size(1) // features.size(1)
            extra = padding_mask.size(1) % features.size(1) # should be 0 since 1st CNN reduces number of features [scale] times (due to the architecture choice)
            assert extra == 0
            padding_mask = padding_mask[:, ::scale]
            assert np.all(padding_mask.shape == features.shape[:-1])
        else:
            scale_float = float(source.size(1)) / features.size(1)

        # TODO maybe move logging segments images to a script also and here just log borders array? or option both here and as a script potentially
        if self.need_logging:
            for i in range(source.shape[0]):
                if self.check_if_log_for_id(id=id[i].item()):
                    if self.repr_data_log_dir:
                        self.log_repr_nonsegmentation_data(source[i], features[i], id=id[i].item() if id is not None else None, epoch=epoch)
                        # [!] logging here, before projection as this is what representation segmentation uses 
                        # - TODO would otherwise need to change unmasked_features = features.clone() to be after projection instead of before

        if self.segm:
            if self.hier_segm_guess_orig:
                unmasked_features = features.clone()  # if guessing original features, get them before averaging
            features, padding_mask, segment_borders, rounding_loss = self.segmentation(features, padding_mask, 5)
            # [!] minSegmsPerLine needs to be at least a few so that part with masking with at least 2 masks works correctly

        if self.need_logging:
            for i in range(source.shape[0]):
                if self.check_if_log_for_id(id=id[i].item()):
                    if self.segm_log_dir:
                        assert self.segm
                        # changed segment lines to only log begins (1 is there now for every segment, -1 if length > 1)
                        # as can just mult by scale for begins, for ends would need to also add scale - 1
                        self.log_named_segmented_image(source[i], [int(round(j*scale_float)) for j, k in enumerate(segment_borders[i]) if k.item() == 1], id=id[i].item() if id is not None else None, epoch=epoch)
                    if self.repr_data_log_dir and self.segm:
                        self.log_repr_segmentation_data(segment_borders[i], id=id[i].item() if id is not None else None, epoch=epoch)
                        # [!] logging here, before projection as this is what representation segmentation uses 
                        # - TODO would otherwise need to change unmasked_features = features.clone() to be after projection instead of before

        if not self.segm or not self.hier_segm_guess_orig:
            unmasked_features = features.clone()

        assert(unmasked_features is not None)

        # doing it here as needed to clone features after segmentation and clone was before post_extract_proj 
        # - [!] TODO maybe check if this (cloning before post_extract_proj) is intended, but perhaps not a very big difference (only linear projection)
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
            q = self.quantizer(y, produce_targets=False)
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
        if self.segm and self.hier_segm_rounding_loss is not None:
            result["rounding_loss"] = rounding_loss

        return result

    def segmentation(self, features, padding_mask, minSegmsPerLine):
        assert self.segm == 'var'  # for now only that supported, to be extended
        non_padded = padding_mask.numel() - padding_mask.sum().item()
        if self.hier_segm_strict_reduction is not None:
            base_len_sum = int(round(non_padded / self.hier_segm_strict_reduction))
            return HierarchicalSegmentationLayer.apply(features, padding_mask, base_len_sum, None, minSegmsPerLine, self.hier_segm_merge_priority, self.hier_segm_shortening_policy, self.hier_segm_rounding_loss)
        else:
            min_reduction, max_reduction = self.hier_segm_reduction_range
            min_segm = base_len_sum = int(round(non_padded / max_reduction))  #max(features.shape[0], int(round(0.85*base_len_sum)))
            max_segm = base_len_sum = int(round(non_padded / min_reduction))  #min(non_padded, int(round(1.15*base_len_sum)))
            return HierarchicalSegmentationLayer.apply(features, padding_mask, None, (min_segm, max_segm), minSegmsPerLine, self.hier_segm_merge_priority, self.hier_segm_shortening_policy, self.hier_segm_rounding_loss)

    def log_segmented_image(self, img, borders, name=None, convert_numbers_from_01=True):
        converted_grayscale_img = img*255. if convert_numbers_from_01 else img
        if torch.is_tensor(converted_grayscale_img):
            converted_grayscale_img = converted_grayscale_img.detach().cpu()
        img = Image.fromarray(np.array(converted_grayscale_img, dtype=np.int32)).convert('RGB')
        draw = ImageDraw.Draw(img)
        for border in borders:
            #if borders[i] != 0:
            #print("!", source[0].shape, i*scale_float)
            draw.line([(border, 0), (border, 31)], fill='red', width=2)
        save_name = name if name is not None else "<random_name_" + str(int(random.random() * 10000000)) + ">"
        img.save(self.segm_log_dir + "/" + save_name + ".png")

    def log_named_segmented_image(self, img, borders, id=None, epoch=None):
        name = "segm_id_" + str(id) + "_epoch_" + str(epoch) if id is not None else None  # will have names with id, possibly overwriting each epoch, otherwise random ids
        self.log_segmented_image(img, borders, name=name, convert_numbers_from_01=True)

    def log_repr_nonsegmentation_data(self, img, features, id=None, epoch=None):
        if torch.is_tensor(img):
            img = img.detach().cpu()
        if torch.is_tensor(features):
            features = features.detach().cpu()
        img_np = np.array(img)
        features_np = np.array(features)
        img_name = "input_id_" + str(id) + "_epoch_" + str(epoch) if id is not None else None  # will have names with id, possibly overwriting each epoch, otherwise random ids
        features_name = "features_id_" + str(id) + "_epoch_" + str(epoch) if id is not None else None  # will have names with id, possibly overwriting each epoch, otherwise random ids
        np.save(self.repr_data_log_dir + "/" + img_name, img_np)
        np.save(self.repr_data_log_dir + "/" + features_name, features_np)

    def log_repr_segmentation_data(self, borders, id=None, epoch=None):
        if torch.is_tensor(borders):
            borders = borders.detach().cpu()
        borders_np = np.array(borders)
        borders_name = "segmentborders_id_" + str(id) + "_epoch_" + str(epoch) if id is not None else None  # will have names with id, possibly overwriting each epoch, otherwise random ids
        np.save(self.repr_data_log_dir + "/" + borders_name, borders_np)
        
    def check_if_log_for_id(self, id=None):
        if self.random_log_freq is not None:
            if random.random() < self.random_log_freq:
                return True
        if self.log_ids is not None:
            assert id is not None  # need to use pass-metadata arg in criterion (if wav2vec, if other need to add this option)
            for log_rule in self.log_ids:
                if log_rule(id):  # check if fits
                    return True
        return False

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

        if self.segm and self.hier_segm_rounding_loss is not None:
            pen.append(net_output["rounding_loss"])

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

        in_d = 1
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

        # BxHxW -> BxCxWxH
        x = x.unsqueeze(1).transpose(-2, -1).contiguous()

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


@register_model_architecture("wav2vec2_scribblelens", "wav2vec2_scribblelens")
def base_architecture(args):
    args.extractor_mode = getattr(args, "extractor_mode", "default")

    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)

    args.final_dim = getattr(args, "final_dim", 0)

    args.layer_norm_first = getattr(args, "layer_norm_first", False)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)

    conv_feature_layers = "[(512, 10, 5)]"
    conv_feature_layers += " + [(512, 8, 4)]"
    conv_feature_layers += " + [(512, 4, 2)] * 3"
    conv_feature_layers += " + [(512, 1, 1)]"
    args.conv_feature_layers = getattr(args, "conv_feature_layers", conv_feature_layers)

    args.logit_temp = getattr(args, "logit_temp", 0.1)

    args.quantize_targets = getattr(args, "quantize_targets", False)
    args.quantize_input = getattr(args, "quantize_input", False)
    args.same_quantizer = getattr(args, "same_quantizer", False)

    args.feature_grad_mult = getattr(args, "feature_grad_mult", 1.0)

    args.latent_vars = getattr(args, "latent_vars", 320)
    args.latent_groups = getattr(args, "latent_groups", 2)
    args.latent_dim = getattr(args, "latent_dim", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.65)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_min_space = getattr(args, "mask_min_space", 1)

    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)
    args.mask_channel_min_space = getattr(args, "mask_channel_min_space", 1)

    args.dropout_input = getattr(args, "dropout_input", 0)
    args.dropout_features = getattr(args, "dropout_features", 0)

    args.num_negatives = getattr(args, "num_negatives", 100)
    args.negatives_from_everywhere = getattr(args, "negatives_from_everywhere", False)
    args.cross_sample_negatives = getattr(args, "cross_sample_negatives", 0)
    args.codebook_negatives = getattr(args, "codebook_negatives", 0)

    args.conv_pos = getattr(args, "conv_pos", 128)
    args.conv_pos_groups = getattr(args, "conv_pos_groups", 16)

    args.latent_temp = getattr(args, "latent_temp", "(2,0.5,0.999995)")

    args.target_glu = getattr(args, "target_glu", False)

    args.conv_bias = getattr(args, "conv_bias", False)
