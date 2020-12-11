#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings and vocab for Scribblelens
"""

import argparse
import os
import json

from fairseq.data.handwriting.scribblelens import ScribbleLensDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--vocab-dir", required=True)
    args = parser.parse_args()

    # embeddings
    os.makedirs(args.output_dir, exist_ok=True)
    
    dd = ScribbleLensDataset(
        root=os.path.join(args.data_dir, 'scribblelens.corpus.v1.2.zip'),           # Path to zip with images
        alignment_root=os.path.join(args.data_dir, 'scribblelens.paths.1.4b.zip'),  # Path to the alignments, i.e. info aou char boundaries
        slice='tasman',                                     # Part of the data, here a single scribe https://en.wikipedia.org/wiki/Abel_Tasman
        split=args.output_name,                                      # Train, test, valid or unsupervised. Train/Test/Valid have character transcripts, unspuervised has only images
        transcript_mode=5,                                  # Legacy space handling, has to be like that
        vocabulary=args.vocab_dir,  # Path
    )

    with open(
        os.path.join(args.output_dir, args.output_name + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.output_dir, args.output_name + ".wrd"), "w"
    ) as wrd_out:
        for elem in dd:
            words = elem['alignment_text'].upper()
            letters = " ".join(words).replace("  ", " |") + " |"
            print(words, file=wrd_out)
            print(letters, file=ltr_out)

    # vocab
    stop_tokens = set(["{", "~", "}", "@"])
    vocab_new = set()
    vocab_count = {}
    with open(
        args.vocab_dir, "r"
    ) as vcb_in, open(
        os.path.join(args.output_dir, "dict.ltr.txt"), "w"
    ) as vcb_out:
        vocab = json.load(vcb_in)
        for tok in vocab:
            if tok not in stop_tokens:
                vocab_new.add(tok.upper())
        vocab_new.add('|')

        for elem in dd:
            words = elem['alignment_text'].upper()
            for tok in words:
                if tok in vocab_new:
                    if tok not in vocab_count:
                        vocab_count[tok] = 1
                    else:
                        vocab_count[tok] += 1
        
        for tok in sorted(vocab_count):
            print(tok, vocab_count[tok], file=vcb_out)
        
        
if __name__ == "__main__":
    main()
