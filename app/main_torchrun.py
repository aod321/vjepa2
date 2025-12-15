#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pprint
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed
from src.utils.logging import get_logger

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument(
    "--local-rank",
    type=int,
    default=0,
    help="Local rank for distributed training (automatically set by torchrun)",
)

def main():
    args = parser.parse_args()
    
    # Set up logging
    logger = get_logger(force=True)
    
    # Get local rank from environment if not provided
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    
    # Set CUDA device for this process
    torch_rank = int(os.environ.get('RANK', local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Set the CUDA device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        logger.info(f"Set CUDA device to: {local_rank}")
    
    # Configure logging level
    if torch_rank == 0:
        import logging
        logger.setLevel(logging.INFO)
    else:
        import logging
        logger.setLevel(logging.ERROR)
    
    logger.info(f"called-params {args.fname}")
    logger.info(f"Local rank: {local_rank}, Global rank: {torch_rank}, World size: {world_size}")
    
    # Load config
    params = None
    with open(args.fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info("loaded params...")
    
    # Log config (only on rank 0)
    if torch_rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = params["folder"]
        params_path = os.path.join(folder, "params-pretrain.yaml")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            yaml.dump(params, f)
    
    # Initialize distributed training
    # The init_distributed function will use the environment variables set by torchrun
    world_size, rank = init_distributed()
    logger.info(f"Running... (rank: {rank}/{world_size})")
    
    # Launch the app with loaded config
    app_main(params["app"], args=params)

if __name__ == "__main__":
    main()
