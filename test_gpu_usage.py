#!/usr/bin/env python3

import torch
import time
import numpy as np
import subprocess

# 导入修改后的模块
from notebooks.utils.world_model_wrapper import WorldModel
from notebooks.utils.mpc_utils import poses_to_diff
from app.vjepa_droid.transforms import make_transforms

def monitor_gpu_memory(device_id=7):
    """监控GPU内存使用"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            f'--id={device_id}',
            '--query-gpu=memory.used,utilization.gpu', 
            '--format=csv,nounits,noheader'
        ], capture_output=True, text=True)
        memory_used, gpu_util = result.stdout.strip().split(', ')
        return int(memory_used), int(gpu_util)
    except:
        return 0, 0

def test_gpu_acceleration():
    print("=== GPU加速测试 ===")
    
    # 检查CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    
    # 加载模型
    print("\n加载模型...")
    encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
    
    # 移到GPU
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    
    print(f"Encoder device: {next(encoder.parameters()).device}")
    print(f"Predictor device: {next(predictor.parameters()).device}")
    
    # 初始化transforms
    crop_size = 256
    tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.),
        reprob=0.,
        auto_augment=False,
    )
    
    # 加载测试数据
    print("\n加载测试数据...")
    trajectory = np.load("notebooks/franka_example_traj.npz")
    np_clips = trajectory["observations"]
    np_states = trajectory["states"]
    
    clips = transform(np_clips[0]).unsqueeze(0).to(device)
    states = torch.tensor(np_states).to(device)
    
    # 编码
    B, C, T, H, W = clips.size()
    clips_reshaped = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(clips_reshaped)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    h = torch.nn.functional.layer_norm(h, (h.size(-1),))
    
    z_n, z_goal = h[:, :tokens_per_frame], h[:, -tokens_per_frame:]
    s_n = states[:, :1]
    
    print(f"Context shape: {z_n.shape}, device: {z_n.device}")
    print(f"Goal shape: {z_goal.shape}, device: {z_goal.device}")
    print(f"State shape: {s_n.shape}, device: {s_n.device}")
    
    # 测试两种配置
    configs = [
        {"force_gpu": False, "name": "Original (with CPU transfer)"},
        {"force_gpu": True, "name": "GPU optimized"}
    ]
    
    for config in configs:
        print(f"\n=== 测试: {config['name']} ===")
        
        # 创建世界模型
        world_model = WorldModel(
            encoder=encoder,
            predictor=predictor,
            tokens_per_frame=tokens_per_frame,
            transform=transform,
            mpc_args={
                "rollout": 2,
                "samples": 10,  # 小一些以便快速测试
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
            device=device,
            force_gpu=config["force_gpu"]
        )
        
        # 监控GPU使用前
        mem_before, util_before = monitor_gpu_memory()
        print(f"GPU状态 (开始): Memory={mem_before}MB, Util={util_before}%")
        
        # 计时运行
        start_time = time.time()
        
        with torch.no_grad():
            actions = world_model.infer_next_action(z_n, s_n, z_goal)
        
        end_time = time.time()
        
        # 监控GPU使用后
        mem_after, util_after = monitor_gpu_memory()
        print(f"GPU状态 (结束): Memory={mem_after}MB, Util={util_after}%")
        
        print(f"运行时间: {end_time - start_time:.2f}秒")
        print(f"结果形状: {actions.shape}")
        
        # 计算与GT的误差
        gt_action = poses_to_diff(np_states[0, 0], np_states[0, 1])
        cem_action = actions[0] if len(actions.shape) == 2 else actions[0, 0]
        if isinstance(cem_action, torch.Tensor):
            cem_action = cem_action.cpu().numpy()
        if isinstance(gt_action, torch.Tensor):
            gt_action = gt_action.cpu().numpy()
        
        error = np.linalg.norm(cem_action - gt_action)
        print(f"L2 Error: {error:.4f}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_gpu_acceleration()