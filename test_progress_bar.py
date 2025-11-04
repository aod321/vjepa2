#!/usr/bin/env python3

import sys
sys.path.insert(0, "..")

import numpy as np
import torch
from torch.nn import functional as F

from app.vjepa_droid.transforms import make_transforms
from notebooks.utils.mpc_utils import poses_to_diff
from notebooks.utils.world_model_wrapper import WorldModel

print("=== 测试进度条功能 ===")

# 加载模型
print("加载模型...")
encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")

device = "cuda:7"
encoder = encoder.to(device)
predictor = predictor.to(device)

# 准备数据
crop_size = 256
tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)
transform = make_transforms(
    random_horizontal_flip=False,
    random_resize_aspect_ratio=(1., 1.),
    random_resize_scale=(1., 1.),
    reprob=0.,
    auto_augment=False,
    motion_shift=False,
    crop_size=crop_size,
)

trajectory = np.load("notebooks/franka_example_traj.npz")
np_clips = trajectory["observations"]
np_states = trajectory["states"]

clips = transform(np_clips[0]).unsqueeze(0).to(device)
states = torch.tensor(np_states).to(device)

def forward_target(c, normalize_reps=True):
    B, C, T, H, W = c.size()
    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h

# 编码
with torch.no_grad():
    h = forward_target(clips)
    z_n, z_goal = h[:, :tokens_per_frame], h[:, -tokens_per_frame:]
    s_n = states[:, :1]

# 创建世界模型
world_model = WorldModel(
    encoder=encoder,
    predictor=predictor,
    tokens_per_frame=tokens_per_frame,
    transform=transform,
    mpc_args={
        "rollout": 2,
        "samples": 8,  # 较小值以便快速看到进度条
        "topk": 4,
        "cem_steps": 50,  # 50步足够看到进度条
        "momentum_mean": 0.15,
        "momentum_mean_gripper": 0.15,
        "momentum_std": 0.75,
        "momentum_std_gripper": 0.15,
        "maxnorm": 0.15,
        "verbose": False  # 关闭verbose，让进度条更清晰
    },
    normalize_reps=True,
    force_gpu=True,
    device=device
)

print("\n=== 测试1: 显示进度条 ===")
with torch.no_grad():
    actions1 = world_model.infer_next_action(z_n, s_n, z_goal, show_progress=True)

print(f"\nCEM完成! Actions shape: {actions1.shape}")

print("\n=== 测试2: 隐藏进度条 ===")
with torch.no_grad():
    actions2 = world_model.infer_next_action(z_n, s_n, z_goal, show_progress=False)

print(f"\nCEM完成! Actions shape: {actions2.shape}")

print("\n测试完成！进度条应该只在第一个测试中显示。")