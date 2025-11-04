# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import torch
import torch.distributed as dist

from src.utils.logging import get_logger

logger = get_logger()


def init_distributed(port=37129, rank_and_world_size=(None, None)):
    # try to set all environment variables to avoid triggering a segfault
    # environment variables can be reallocated during the execution of torch.distributed.init_process_group
    # the idea is a race condition may trigger if init_progress_group is modifying an environment variable at
    # the same time as Python, so we try to set all environs before initializing distributed
    if "SLURM_JOB_ID" in os.environ:
        # Use the slurm_tmpdir (if it exists) instead of /tmp
        tmpdir = Path(f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}")
        if tmpdir.exists():
            os.environ["TMPDIR"] = str(tmpdir)

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    
    # First priority: Use torchrun environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # torchrun already sets MASTER_ADDR and MASTER_PORT
        logger.info(f"Using torchrun env vars: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    elif (rank is None) or (world_size is None):
        # Second priority: Use SLURM environment variables
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
            os.environ["MASTER_PORT"] = str(port)
            logger.info(f"Using SLURM env vars: rank={rank}, world_size={world_size}")
        except Exception:
            logger.info("SLURM vars not set (distributed training not available)")
            world_size, rank = 1, 0
            return world_size, rank
    else:
        # Third priority: Use provided rank_and_world_size
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        logger.info(f"Using provided values: rank={rank}, world_size={world_size}")

    try:
        # Initialize process group even for single process to support DDP in debug mode
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, world_size=world_size, rank=rank)
        if world_size > 1:
            logger.info(f"Successfully initialized distributed training: rank {rank}/{world_size}")
        else:
            logger.info(f"Single process training with DDP support: rank {rank}/{world_size}")
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f"Rank: {rank}. Distributed training not available {e}")

    return world_size, rank


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
