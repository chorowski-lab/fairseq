# Copyright Facebook, Inc. and its affiliates., UWr 2021
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging
import os
import sys
import numpy as np

import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as tk
import torchaudio.functional as AF

from .. import FairseqDataset
from . import raw_audio_dataset

logger = logging.getLogger(__name__)

class SpeechFeaturesDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        transform='fbank',
        transform_kwargs=None,
        add_deltas=True,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        pad_to_multiples_of=None,
        normalize=False,
    ):
        super().__init__()
        if transform_kwargs is None:
            transform_kwargs = {}
        else:
            transform_kwargs = dict(transform_kwargs)
        
        if transform_kwargs.get('sample_frequency', sample_rate) != sample_rate:
            logging.warn("Reseting sample frequency")
        transform_kwargs['sample_frequency'] = sample_rate
        transform_kwargs.setdefault('frame_shift', 10.0)
        stride = sample_rate * transform_kwargs['frame_shift'] * 1e-3

        self.fnames = []
        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.min_length = min_length
        self.pad = pad
        self.pad_to_multiples_of = pad_to_multiples_of
        self.add_deltas = add_deltas
        self.shuffle = shuffle
        self.normalize = normalize

        skipped = 0
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for line in f:
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(math.ceil(int(items[1]) / stride))
                if min_length is not None and sz < min_length:
                    skipped += 1
                    continue
                self.fnames.append(items[0])
                self.sizes.append(sz)
        logger.info(f"{manifest_path}: loaded {len(self.fnames)}, skipped {skipped} samples")

        self.transform = getattr(tk, transform)
        self.transform_kwargs = transform_kwargs

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats):
        if self.normalize:
            with torch.no_grad():
                # Normalize only across time
                feats = F.layer_norm(feats, feats.shape[-1:])
        return feats

    
    def __getitem__(self, index):
        import soundfile as sf

        fname = os.path.join(self.root_dir, self.fnames[index])
        wav, curr_sample_rate = sf.read(fname)
        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        feats = torch.from_numpy(wav).float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        feats = feats.unsqueeze(0)  # C=1,L

        feats = self.transform(feats, **self.transform_kwargs)
        feats = feats.transpose(0, 1)  # Nmels  x Len
        feats = feats.unsqueeze(0)  # 1 x Nmels x Len

        if self.add_deltas:
            deltas = AF.compute_deltas(feats)
            ddeltas = AF.compute_deltas(deltas)
            feats = torch.cat([feats, deltas, ddeltas], dim=0)

        feats = self.postprocess(feats)
        return {"id": index, "source": feats}

    def crop_to_max_size(self, wav, target_size):
        size = wav.shape[2]
        diff = size - target_size
        if diff <= 0:
            return wav

        if self.shuffle:
            start = np.random.randint(0, diff + 1)
        else:
            # Deterministically pick the middle part
            start = (diff + 1) //2
        end = size - diff + start
        return wav[:, :, start:end]

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [s.shape[2] for s in sources]
        heigths = [s.shape[1] for s in sources]
        chans = [s.shape[0] for s in sources]
        
        assert all([h==heigths[0] for h in heigths])
        assert all([h==chans[0] for h in chans])

        pad_to_multiples_of = self.pad_to_multiples_of
        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
            if pad_to_multiples_of:
                # round up to pad_to_multiples_of
                target_size = ((target_size + pad_to_multiples_of - 1) // pad_to_multiples_of) * pad_to_multiples_of
        else:
            target_size = min(min(sizes), self.max_sample_size)
            if pad_to_multiples_of:
                # round down to pad_to_multiples_of
                target_size = (target_size // pad_to_multiples_of) * pad_to_multiples_of

        collated_sources = sources[0].new_zeros((len(sources), chans[0], heigths[0], target_size))
        pad_shape = [len(sources), 1, target_size]
        padding_mask = (
            torch.BoolTensor(size=pad_shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((chans[0], heigths[0], -diff,), 0.0)],
                    dim=2
                )
                padding_mask[i, :, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask
        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input}

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]