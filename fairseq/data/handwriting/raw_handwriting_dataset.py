# Copyright (c) UWr and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import logging
import numpy as np
import sys

import torch
import torch.nn.functional as F

from . import scribblelens
from .. import FairseqDataset


logger = logging.getLogger(__name__)


class RawHandwritingDataset(FairseqDataset):
    def __init__(
        self,
        max_sample_size=None,
        min_sample_size=None,
        pad_to_multiples_of=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
        labels=False,  # [!] if True, need to set pad, blank and eos indices (set_special_indices)
    ):
        super().__init__()

        # We don't really have a sampling rate - out of audio (JCh)
        # self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.min_length = min_length
        self.pad_to_multiples_of = pad_to_multiples_of
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize
        self.labels = labels
        self.label_pad_idx = None
        self.label_blank_idx = None
        self.label_eos_idx = None

    def set_special_indices(
        self,
        label_pad_idx,
        label_blank_idx,  # to ignore in alignment when getting cropped label etc
        decoder_fun,
        label_eos_idx=None,  # leave None for not appending EOS
    ):
        self.label_pad_idx = label_pad_idx
        self.label_blank_idx = label_blank_idx
        self.label_eos_idx = label_eos_idx

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        # TODO(jch): verify if this makes sense, prob not!
        # if feats.dim() == 2:
        #     feats = feats.mean(-1)

        # # Doesn't make sense - JCh
        # # if curr_sample_rate != self.sample_rate:
        # #     raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        # assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, wav, target_size_dim1, alignment=None):

        # if alignment set, cut it too - TODO maybe also mask half a letter etc., also in data!

        size = wav.shape[1] #len(wav)
        diff = size - target_size_dim1
        if diff <= 0:
            if alignment:
                return wav, alignment
            else:
                return wav

        if self.shuffle:
            start = np.random.randint(0, diff + 1)
        else:
            # Deterministically pick the middle part
            start = (diff + 1) //2
        end = size - diff + start
        if alignment:
            return wav[:, start:end], alignment[start:end]
        else:
            return wav[:, start:end]
        
    def collater(self, samples):

        # TODO stuff with labels
        # collated = self.dataset.collater(samples)
        # if len(collated) == 0:
        #     return collated
        # indices = set(collated["id"].tolist())
        # target = [s["label"] for s in samples if s["id"] in indices]

        # if self.batch_targets:
        #     collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
        #     target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
        #     collated["ntokens"] = collated["target_lengths"].sum().item()
        # else:
        #     collated["ntokens"] = sum([len(t) for t in target])

        # collated["target"] = target

        # if self.add_to_input:
        #     eos = target.new_full((target.size(0), 1), self.eos)
        #     collated["target"] = torch.cat([target, eos], dim=-1).long()
        #     collated["net_input"]["prev_output_tokens"] = torch.cat([eos, target], dim=-1).long()
        #     collated["ntokens"] += target.size(0)
        #return collated



        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [s.shape[1] for s in sources]
        heigths = [s.shape[0] for s in sources]
        assert all([h==heigths[0] for h in heigths])

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

        collated_sources = sources[0].new_zeros((len(sources), heigths[0], target_size))
        pad_shape = list(collated_sources.shape)
        pad_shape[1] = 1  # we mask all pixels in exactly the same way
        padding_mask = (
            torch.BoolTensor(size=pad_shape).fill_(False) if self.pad else None
        )
        if self.labels:
            collated_labels_nontensor = []
            #collated_texts_nontensor = []  # TODO
            collated_alignments = samples[0]["alignment"].new_zeros((len(sources), target_size))

        for i, (sample, size) in enumerate(zip(samples, sizes)):
            source = sample["source"]
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
                if self.labels:
                    collated_labels_nontensor.append(sample["label"])
                    #collated_texts_nontensor.append(sample["text"])
                    collated_alignments[i] = sample["alignment"]
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((heigths[0], -diff), 0.0)],
                    dim=1
                )
                padding_mask[i, :, diff:] = True
                if self.labels:
                    collated_alignments[i] = torch.cat([sample["alignment"], sample["alignment"].new_full((-diff,), self.label_pad_idx)])
                    coll_labels = sample["label"]  #self.collate_labels(collated_alignments[i], sample["label"], sample["text"])
                    collated_labels_nontensor.append(coll_labels)
                    #collated_texts_nontensor.append(coll_text)
            else:
                # only case with cropping  TODO fix case with double letters without space between
                if self.labels:
                    collated_sources[i], collated_alignments[i] = self.crop_to_max_size(source, target_size, alignment=sample["alignment"])
                    coll_labels = self.collate_labels(collated_alignments[i], sample["label"], sample["text"])
                    collated_labels_nontensor.append(coll_labels)
                    #collated_texts_nontensor.append(coll_text)
                else:
                    collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask
        if self.labels:
            assert self.pad
            collated_labels = torch.IntTensor(size=(len(collated_labels_nontensor), max([len(i) for i in collated_labels_nontensor]))).fill_(self.label_pad_idx)
            for i, label in enumerate(collated_labels_nontensor):
                collated_labels[i][:len(label)] = torch.tensor(label)
            # TODO check collate labels to common length in a tensor
            # TODO EOS stuff (?)
            target_lengths = torch.LongTensor([len(t) for t in collated_labels_nontensor])
            input["alignments"] = collated_alignments
            return {
                "id": torch.LongTensor([s["id"] for s in samples]), 
                "net_input": input,
                "target_lengths": target_lengths,
                "target": collated_labels,  # data_utils.collate_tokens(collated_labels_nontensor, pad_idx=self.pad, left_pad=False),
                "ntokens": target_lengths.sum().item(),
                "alignments": collated_alignments
                #"label_texts": collated_texts_nontensor,  # TODO?  non-collated texts of collated stuff
                }
        else:
            return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input}

    def collate_labels(self, collated_alignments, full_label, full_text):  # label is a list, text is a string
        last_idx = self.label_blank_idx
        #decode_dict = {x.item(): y for x,y in zip(full_label, full_text)}  # can zip like that as full_label is already a list
        collated_label = []
        # [!] TODO fix case with double letters and stuff
        for num in collated_alignments:
            if num.item() != last_idx and num.item() != self.label_pad_idx:
                last_idx = num
                if num.item() != self.label_blank_idx:
                    collated_label.append(num.item())
        #collated_text = ''.join([decode_dict[x] for x in collated_label])
        return collated_label #, collated_text


    def num_tokens(self, index):
        return self.size(index)  # TODO this doesn't really seem correct if tokens are letters as I think

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)
        
        # TODO stuff with labels? in addTargetDataset there is a 2nd dim then

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)  # TODO should return also label size with labels? (as in AddTargetDataset), but this screws up much other stuff
        return np.lexsort(order)[::-1]


class FileHandwritingDataset(RawHandwritingDataset):
    def __init__(
        self,
        dist_root,
        split,
        max_sample_size=None,
        min_sample_size=None,
        pad_to_multiples_of=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
        labels=False,
    ):
        super().__init__(
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            pad_to_multiples_of=pad_to_multiples_of,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
            labels=labels,
        )
        self.dataset = scribblelens.ScribbleLensDataset(
            root=dist_root + '/scribblelens.corpus.v1.2.zip',           # Path to zip with images
            alignment_root=dist_root + '/scribblelens.paths.1.4b.zip',  # Path to the alignments, i.e. info aou char boundaries
            slice='tasman',                                     # Part of the data, here a single scribe https://en.wikipedia.org/wiki/Abel_Tasman
            split=split,                                      # Train, test, valid or unsupervised. Train/Test/Valid have character transcripts, unspuervised has only images
            # Not used in the simple ScribbleLens loader
            transcript_mode=5,                                  # Legacy space handling, has to be like that
            vocabulary=FileHandwritingDataset.vocabularyPath(dist_root),  # Path
        )
        if labels:
            self.set_special_indices(
                self.dataset.alphabet.pad(),
                self.dataset.alphabet.blank(),
                self.dataset.alphabet.eos()
            )
        # self.labels in superclass

        for data in self.dataset:
            sizeHere = data['image'].shape
            #print(sizeHere)
            #if self.labels:
            #    self.sizes.append((sizeHere[0], data['text'].shape[0]))  # ?
            #else:
            # not sure why AddTargetDataset has label size in sizes and makes it a tuple, of course not a single comment and incompatible because why anything would be 
            self.sizes.append(sizeHere[0])  # 1/2 dim TODO? rather this 1 dim is correct

        # self.fnames = []

        # skipped = 0
        # with open(manifest_path, "r") as f:
        #     self.root_dir = f.readline().strip()
        #     for line in f:
        #         items = line.strip().split("\t")
        #         assert len(items) == 2, line
        #         sz = int(items[1])
        #         if min_length is not None and sz < min_length:
        #             skipped += 1
        #             continue
        #         self.fnames.append(items[0])
        #         self.sizes.append(sz)
        # logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

    @staticmethod
    def vocabularyPathSuffix():
        return '/tasman.alphabet.plus.space.mode5.json'

    @staticmethod
    def vocabularyPath(prefix):
        return prefix + FileHandwritingDataset.vocabularyPathSuffix()

    def __getitem__(self, index):
        # import soundfile as sf

        # fname = os.path.join(self.root_dir, self.fnames[index])
        # wav, curr_sample_rate = sf.read(fname)
        # feats = torch.from_numpy(wav).float()
        # feats = self.postprocess(feats, curr_sample_rate)

        feats = self.dataset[index]['image'][:,:,0]

        if self.labels:
            return {
                "id": index, 
                "source": feats.T,  # image 32 x W
                "alignment": self.dataset[index]['alignment'],
                "label": self.dataset[index]['text'],
                "text": self.dataset[index]['alignment_text']
            }
        else:
            return {"id": index, "source": feats.T}