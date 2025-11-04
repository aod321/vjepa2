#!/usr/bin/env python3

import sys
sys.path.insert(0, "..")

import numpy as np
import torch
from torch.nn import functional as F
import subprocess
import time

from app.vjepa_droid.transforms import make_transforms
from notebooks.utils.mpc_utils import poses_to_diff
from notebooks.utils.world_model_wrapper import WorldModel

def monitor_gpu(gpu_id=7):
    """检查GPU使用情况"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            f'--id={gpu_id}',
            '--query-gpu=memory.used,utilization.gpu', 
            '--format=csv,nounits,noheader'
        ], capture_output=True, text=True)
        memory_used, gpu_util = result.stdout.strip().split(', ')
        return int(memory_used), int(gpu_util)
    except:
        return 0, 0

print("=== 测试修改后的worldmodel代码 ===")

# 1. 检查初始GPU状态
mem_before, util_before = monitor_gpu(7)
print(f"初始GPU 7状态: Memory={mem_before}MB, Util={util_before}%")

# 2. 加载模型
print("\n加载模型...")
encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")

# 移动模型到GPU
device = "cuda:7"
encoder = encoder.to(device)
predictor = predictor.to(device)
print(f"Models moved to {device}")
print(f"Encoder device: {next(encoder.parameters()).device}")
print(f"Predictor device: {next(predictor.parameters()).device}")

mem_after_load, util_after_load = monitor_gpu(7)
print(f"加载模型后GPU 7状态: Memory={mem_after_load}MB, Util={util_after_load}%")

# 3. 准备数据
print("\n准备数据...")
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

# Convert trajectory to torch tensors (移动到GPU)
clips = transform(np_clips[0]).unsqueeze(0).to(device)
states = torch.tensor(np_states).to(device)

print(f"clips: {clips.shape}, device: {clips.device}")
print(f"states: {states.shape}, device: {states.device}")

# 4. 编码
print("\n编码数据...")
def forward_target(c, normalize_reps=True):
    B, C, T, H, W = c.size()
    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h

with torch.no_grad():
    h = forward_target(clips)
    z_n, z_goal = h[:, :tokens_per_frame], h[:, -tokens_per_frame:]
    s_n = states[:, :1]

print(f"z_n device: {z_n.device}, shape: {z_n.shape}")
print(f"z_goal device: {z_goal.device}, shape: {z_goal.shape}")
print(f"s_n device: {s_n.device}, shape: {s_n.shape}")

mem_after_encode, util_after_encode = monitor_gpu(7)
print(f"编码后GPU 7状态: Memory={mem_after_encode}MB, Util={util_after_encode}%")

# 5. 创建世界模型并运行CEM
print("\n创建世界模型...")
world_model = WorldModel(
    encoder=encoder,
    predictor=predictor,
    tokens_per_frame=tokens_per_frame,
    transform=transform,
    mpc_args={
        "rollout": 2,
        "samples": 10,  # 小一些便于测试
        "topk": 5,
        "cem_steps": 20,
        "momentum_mean": 0.15,
        "momentum_mean_gripper": 0.15,
        "momentum_std": 0.75,
        "momentum_std_gripper": 0.15,
        "maxnorm": 0.15,
        "verbose": True
    },
    normalize_reps=True,
    force_gpu=True,
    device=device
)

print("开始CEM优化...")
print("在另一个终端运行: watch -n 1 'gpustat'")
print("=" * 50)

start_time = time.time()
with torch.no_grad():
    actions = world_model.infer_next_action(z_n, s_n, z_goal)

end_time = time.time()
print("=" * 50)

mem_after_cem, util_after_cem = monitor_gpu(7)
print(f"CEM完成，用时: {end_time - start_time:.2f}秒")
print(f"CEM后GPU 7状态: Memory={mem_after_cem}MB, Util={util_after_cem}%")
print(f"Actions shape: {actions.shape}")

# 6. 计算误差
cem_action = actions[0] if len(actions.shape) == 2 else actions[0, 0]
gt_action = poses_to_diff(np_states[0, 0], np_states[0, 1])

if isinstance(cem_action, torch.Tensor):
    cem_action = cem_action.cpu().numpy()
if isinstance(gt_action, torch.Tensor):
    gt_action = gt_action.cpu().numpy()

error = np.linalg.norm(cem_action - gt_action)
print(f"\nL2 error: {error:.4f}")
print(f"CEM action: {cem_action}")
print(f"GT action: {gt_action}")

print("\n测试完成！如果你看到GPU 7的内存和利用率增加，说明GPU被正确使用了。")