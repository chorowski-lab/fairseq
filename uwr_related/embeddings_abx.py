#!/usr/bin/env python3 -u
# !/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import argparse
from itertools import chain
from pathlib import Path
from time import time
import numpy as np
import soundfile as sf

import torch
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from omegaconf import DictConfig


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("zerospeech2021 abx")

def loadCheckpoint(path_checkpoint, path_data):
    """
    Load lstm_lm model from checkpoint.
    """
    # Set up the args Namespace
    model_args = argparse.Namespace(
        task="language_modeling",
        output_dictionary_size=-1,
        data=path_data,
        path=path_checkpoint
        )

    # Setup task
    #task = tasks.setup_task(model_args)
    
    # Load model
    models, _model_args = checkpoint_utils.load_model_ensemble([model_args.path])
    model = models[0]
    
    return model#, task

def parse_args():
    # Run parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("path_checkpoint", type=str,
                        help="Path to the trained fairseq wav2vec2.0 model.")
    parser.add_argument("path_data", type=str,
                        help="Path to the dataset that we want to compute ABX for.")
    parser.add_argument("path_output_dir", type=str,
                        help="Path to the output directory.")
    parser.add_argument("--debug", action="store_true",
                        help="Load only a very small amount of files for "
                        "debugging purposes.")
    parser.add_argument("--cpu", action="store_true",
                        help="Run on a cpu machine.")
    parser.add_argument("--file_extension", type=str, default="wav",
                          help="Extension of the audio files in the dataset (default: wav).")
    return parser.parse_args()

def main():
    # Parse and print args
    args = parse_args()
    logger.info(args)

    # Load the model
    print("")
    print(f"Loading model from {args.path_checkpoint}")
    model = loadCheckpoint(args.path_checkpoint, args.path_data)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    
    # Register the hooks
    layer_outputs = {}
    def get_layer_output(name):
        def hook(model, input, output):
            if type(output) is tuple:
                layer_outputs[name] = output[0].detach().squeeze(1).cpu().numpy()
            elif type(output) is dict:
                layer_outputs[name] = output["x"].detach().squeeze(0).cpu().numpy()
            else:
                layer_outputs[name] = output.detach().squeeze(0).cpu().numpy()
        return hook

    for i in range(len(model.encoder.layers)):
        layer_name = "layer_{}".format(i)
        model.encoder.layers[i].register_forward_hook(get_layer_output(layer_name))
    model.register_forward_hook(get_layer_output("last"))

    model = model.eval().to(device)  
    print("Wav2Vec2 model loaded!")
    print(model)

    # Extract values from chosen layers and save them to files
    phonetic = "phonetic"
    datasets_path = os.path.join(args.path_data, phonetic)
    datasets = os.listdir(datasets_path)
    print(datasets)

    with torch.no_grad():     
        for dataset in datasets:
            print("> {}".format(dataset))
            dataset_path = os.path.join(datasets_path, dataset)
            files = [f for f in os.listdir(dataset_path) if f.endswith(args.file_extension)]
            for i, f in enumerate(files):
                print("Progress {:2.1%}".format(i / len(files)), end="\r")
                input_f = os.path.join(dataset_path, f)
                x, sample_rate = sf.read(input_f)
                x = torch.tensor(x).float().reshape(1,-1).to(device)
                net_output = model(x, features_only=True)

                for layer_name, value in layer_outputs.items():
                    output_dir = os.path.join(args.path_output_dir, layer_name, phonetic, dataset)
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    out_f = os.path.join(output_dir, os.path.splitext(f)[0] + ".txt")
                    np.savetxt(out_f, value)

if __name__ == "__main__":
    main()

