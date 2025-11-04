#!/usr/bin/env python3

import torch
import time
import subprocess
import threading
import numpy as np
from notebooks.utils.world_model_wrapper import WorldModel
from notebooks.utils.mpc_utils import poses_to_diff
from app.vjepa_droid.transforms import make_transforms

def monitor_gpu(stop_event, gpu_id=7, interval=2):
    """持续监控GPU使用情况"""
    print("=== GPU监控开始 ===")
    while not stop_event.is_set():
        try:
            result = subprocess.run([
                'nvidia-smi', 
                f'--id={gpu_id}',
                '--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu', 
                '--format=csv,nounits,noheader'
            ], capture_output=True, text=True)
            
            timestamp, memory_used, gpu_util, temp = result.stdout.strip().split(', ')
            print(f"[GPU {gpu_id}] {timestamp}: Memory={memory_used}MB, Util={gpu_util}%, Temp={temp}°C")
        except:
            print(f"[GPU {gpu_id}] 监控失败")
        
        stop_event.wait(interval)
    print("=== GPU监控结束 ===")

def main():
    print("=== 开始GPU使用测试 ===")
    
    # 启动GPU监控线程
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_gpu, args=(stop_event, 7, 1))
    monitor_thread.start()
    
    try:
        # 设置设备
        device = "cuda:7"
        torch.cuda.set_device(7)
        
        print(f"\n1. 加载模型...")
        encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
        
        print(f"2. 移动模型到 {device}")
        encoder = encoder.to(device)
        predictor = predictor.to(device)
        
        print(f"   Encoder device: {next(encoder.parameters()).device}")
        print(f"   Predictor device: {next(predictor.parameters()).device}")
        print(f"   GPU 7 内存: {torch.cuda.memory_allocated(7) / 1024**2:.1f} MB")
        
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
        
        print(f"\n3. 加载数据...")
        trajectory = np.load("notebooks/franka_example_traj.npz")
        np_clips = trajectory["observations"]
        np_states = trajectory["states"]
        
        clips = transform(np_clips[0]).unsqueeze(0).to(device)
        states = torch.tensor(np_states).to(device)
        
        print(f"   Clips device: {clips.device}, shape: {clips.shape}")
        print(f"   States device: {states.device}, shape: {states.shape}")
        
        print(f"\n4. 编码数据...")
        with torch.no_grad():
            B, C, T, H, W = clips.size()
            clips_reshaped = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
            h = encoder(clips_reshaped)
            h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
            h = torch.nn.functional.layer_norm(h, (h.size(-1),))
        
        z_n, z_goal = h[:, :tokens_per_frame], h[:, -tokens_per_frame:]
        s_n = states[:, :1]
        
        print(f"   编码完成，GPU 7 内存: {torch.cuda.memory_allocated(7) / 1024**2:.1f} MB")
        
        print(f"\n5. 创建世界模型...")
        world_model = WorldModel(
            encoder=encoder,
            predictor=predictor,
            tokens_per_frame=tokens_per_frame,
            transform=transform,
            mpc_args={
                "rollout": 2,
                "samples": 20,  # 较小值以便快速看到效果
                "topk": 8,
                "cem_steps": 30,
                "momentum_mean": 0.15,
                "momentum_mean_gripper": 0.15,
                "momentum_std": 0.75,
                "momentum_std_gripper": 0.15,
                "maxnorm": 0.15,
                "verbose": False  # 减少输出
            },
            normalize_reps=True,
            device=device,
            force_gpu=True
        )
        
        print(f"\n6. 开始CEM优化 (观察GPU监控)...")
        print("=" * 50)
        
        start_time = time.time()
        with torch.no_grad():
            actions = world_model.infer_next_action(z_n, s_n, z_goal)
        end_time = time.time()
        
        print("=" * 50)
        print(f"CEM完成！用时: {end_time - start_time:.2f}秒")
        print(f"结果形状: {actions.shape}")
        print(f"最终GPU 7 内存: {torch.cuda.memory_allocated(7) / 1024**2:.1f} MB")
        
        # 计算误差
        gt_action = poses_to_diff(np_states[0, 0], np_states[0, 1])
        cem_action = actions[0] if len(actions.shape) == 2 else actions[0, 0]
        if isinstance(cem_action, torch.Tensor):
            cem_action = cem_action.cpu().numpy()
        if isinstance(gt_action, torch.Tensor):
            gt_action = gt_action.cpu().numpy()
        
        error = np.linalg.norm(cem_action - gt_action)
        print(f"L2 Error: {error:.4f}")
        
        print("\n等待5秒以观察GPU状态...")
        time.sleep(5)
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 停止监控
        stop_event.set()
        monitor_thread.join()
        print("测试完成")

if __name__ == "__main__":
    main()