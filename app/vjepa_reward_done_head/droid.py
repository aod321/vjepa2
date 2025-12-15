import json
import os
from logging import getLogger
from typing import Optional

import h5py
import numpy as np
import torch
import torch.utils.data
from decord import VideoReader, cpu

logger = getLogger(__name__)


def get_json(directory: str) -> Optional[dict]:
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error decoding JSON in file: {filename}")
            except Exception as e:
                logger.warning(f"Unexpected error while processing {filename}: {e}")
    return None


class RLRolloutDataset(torch.utils.data.Dataset):
    """
    RL rollout dataset with per-step rewards and dones stored alongside video.
    Expects per-episode directories containing `metadata.json`, `trajectory.h5`,
    and the video specified by metadata["video_path"].
    """

    def __init__(
        self,
        data_path: str,
        frames_per_clip: int,
        fps: int = 10,
        sample_stride: int = 1,
        tubelet_size: int = 1,
        transform=None,
        sample_tail_prob: float = 0.0,
        derive_dones: bool = False,
        return_actions: bool = False,
    ) -> None:
        if frames_per_clip <= 0:
            raise ValueError("frames_per_clip must be positive.")
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.sample_stride = max(sample_stride, 1)
        self.tubelet_size = max(tubelet_size, 1)
        self.transform = transform
        self.sample_tail_prob = float(sample_tail_prob)
        self.derive_dones = derive_dones
        self.return_actions = return_actions

        samples = np.loadtxt(data_path, dtype=str, ndmin=1)
        if samples.ndim == 0:
            paths = [str(samples)]
        else:
            paths = [str(s) for s in samples.tolist()]
        # Drop optional header row if present
        if paths and paths[0].lower() == "episode_path":
            paths = paths[1:]
        self.samples = paths

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_video_path(self, root: str, rel: str) -> str:
        if os.path.isabs(rel):
            return rel
        candidate = os.path.join(root, rel)
        if os.path.exists(candidate):
            return candidate
        return candidate  # fall back even if missing; decord will raise

    def __getitem__(self, index: int):
        path = self.samples[index]
        loaded = False
        while not loaded:
            try:
                clip, rewards, dones, actions, indices = self._load_clip(path)
                loaded = True
            except Exception as e:
                logger.warning(f"Failed to load sample {path} ({e}), resampling...")
                index = torch.randint(0, len(self.samples), (1,)).item()
                path = self.samples[index]
        if self.return_actions:
            return clip, rewards, dones, actions, indices
        return clip, rewards, dones, indices

    def _load_clip(self, path: str):
        metadata = get_json(path)
        if metadata is None:
            raise RuntimeError(f"No metadata found at {path}")
        video_rel = metadata.get("video_path")
        if video_rel is None:
            raise RuntimeError(f"No video_path in metadata for {path}")
        tpath = os.path.join(path, "trajectory.h5")
        if not os.path.exists(tpath):
            raise RuntimeError(f"trajectory.h5 missing for {path}")

        with h5py.File(tpath, "r") as traj:
            episode = traj.get("episode")
            if episode is None:
                raise RuntimeError(f"'episode' group missing in {tpath}")
            if "rewards" not in episode:
                raise RuntimeError(f"'rewards' missing in {tpath}")
            rewards = np.array(episode["rewards"], dtype=np.float32)
            if "dones" in episode:
                dones = np.array(episode["dones"], dtype=np.float32)
            elif self.derive_dones:
                dones = (rewards > 0).astype(np.float32)
                if dones.sum() == 0:
                    dones[-1] = 1.0  # mark final step as done if never positive
                # Avoid spamming logs when deriving dones for many episodes
            else:
                raise RuntimeError(f"'dones' missing in {tpath}")
            actions = None
            if self.return_actions:
                if "actions" not in episode:
                    raise RuntimeError(f"'actions' missing in {tpath} but required for policy head")
                actions = np.array(episode["actions"], dtype=np.int64)

        vpath = self._resolve_video_path(path, video_rel)
        vr = VideoReader(vpath, ctx=cpu(0))
        vlen = len(vr)
        if rewards.shape[0] != vlen or dones.shape[0] != vlen or (actions is not None and actions.shape[0] != vlen):
            raise RuntimeError(
                f"Length mismatch video/rewards/dones/actions at {path}: vlen={vlen}, "
                f"rewards={rewards.shape[0]}, dones={dones.shape[0]}, actions={-1 if actions is None else actions.shape[0]}"
            )

        if vlen < self.frames_per_clip:
            raise RuntimeError(f"Video too short at {path}: need {self.frames_per_clip}, got {vlen}")

        # Sample window
        max_start = vlen - self.frames_per_clip
        if max_start < 0:
            raise RuntimeError(f"Cannot sample window from {path}, max_start={max_start}")
        if self.sample_tail_prob > 0 and torch.rand(1).item() < self.sample_tail_prob:
            start = max_start  # focus tail to cover completion steps
        else:
            start = torch.randint(0, max_start + 1, (1,)).item()
        indices = np.arange(start, start + self.frames_per_clip, dtype=np.int64)
        indices = indices[:: self.sample_stride]

        if len(indices) % self.tubelet_size != 0:
            raise RuntimeError(
                f"indices length {len(indices)} is not divisible by tubelet_size {self.tubelet_size} at {path}"
            )

        rewards = rewards[indices]
        dones = dones[indices]
        if actions is not None:
            actions = actions[indices]
        vr.seek(0)
        buffer = vr.get_batch(indices).asnumpy()  # shape [T, H, W, C]
        if self.transform is not None:
            buffer = self.transform(buffer)

        if rewards.shape[0] % self.tubelet_size != 0:
            raise RuntimeError(
                f"Clip/label length not divisible by tubelet_size at {path}: clip={rewards.shape[0]}"
            )
        # Group by tubelet: use the latest frame label in each tubelet
        new_len = rewards.shape[0] // self.tubelet_size
        rewards = rewards.reshape(new_len, self.tubelet_size)[:, -1]
        dones = dones.reshape(new_len, self.tubelet_size)[:, -1]
        if actions is not None:
            actions = actions.reshape(new_len, self.tubelet_size)[:, -1]
        if isinstance(buffer, np.ndarray):
            # buffer shape: T H W C
            if buffer.shape[0] % self.tubelet_size != 0:
                raise RuntimeError(f"Buffer length mismatch after grouping at {path}")
            buffer_len = buffer.shape[0]
        else:
            # torch tensor: C T H W
            if buffer.shape[1] % self.tubelet_size != 0:
                raise RuntimeError(f"Buffer length mismatch after grouping at {path}")
            buffer_len = buffer.shape[1]

        if rewards.shape[0] != (buffer_len // self.tubelet_size):
            raise RuntimeError(
                f"Clip/label length mismatch at {path}: "
                f"clip_tubelets={buffer_len // self.tubelet_size}, labels={rewards.shape[0]}"
            )

        return buffer, rewards, dones, actions, indices


def init_data(
    data_path: str,
    batch_size: int,
    frames_per_clip: int,
    fps: int = 10,
    sample_stride: int = 1,
    tubelet_size: int = 1,
    rank: int = 0,
    world_size: int = 1,
    drop_last: bool = True,
    num_workers: int = 4,
    pin_mem: bool = True,
    persistent_workers: bool = True,
    collator=None,
    transform=None,
    sample_tail_prob: float = 0.0,
    derive_dones: bool = False,
    return_actions: bool = False,
):
    dataset = RLRolloutDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        fps=fps,
        sample_stride=sample_stride,
        tubelet_size=tubelet_size,
        transform=transform,
        sample_tail_prob=sample_tail_prob,
        derive_dones=derive_dones,
        return_actions=return_actions,
    )
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=dist_sampler,
        collate_fn=collator,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    logger.info("RL rollout data loader created")
    return loader, dist_sampler
