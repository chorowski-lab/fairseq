#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Hydra entry point with debugging.
"""

from fairseq_cli.hydra_train import cli_main


if __name__ == '__main__':
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7309))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()
    cli_main()
