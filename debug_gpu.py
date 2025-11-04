#!/usr/bin/env python3

import torch
import time
import subprocess

def check_gpu_usage():
    """检查具体GPU使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', 
                               '--format=csv,nounits,noheader'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        for line in lines:
            gpu_id, name, mem_used, util = line.split(', ')
            print(f"GPU {gpu_id}: {name}, Memory: {mem_used}MB, Util: {util}%")
    except:
        print("无法获取GPU状态")

def debug_model_placement():
    print("=== 调试模型设备放置 ===\n")
    
    # 1. 检查CUDA环境
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("\n=== 加载模型前的GPU状态 ===")
    check_gpu_usage()
    
    # 2. 显式设置设备
    target_device = "cuda:7"
    print(f"\n设置目标设备: {target_device}")
    
    # 设置默认设备
    torch.cuda.set_device(7)
    print(f"默认CUDA设备设置为: {torch.cuda.current_device()}")
    
    # 3. 加载模型并检查
    print("\n=== 加载模型 ===")
    encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
    
    print(f"模型加载后，encoder参数设备: {next(encoder.parameters()).device}")
    print(f"模型加载后，predictor参数设备: {next(predictor.parameters()).device}")
    
    print("\n=== 加载模型后的GPU状态 ===")
    check_gpu_usage()
    
    # 4. 显式移动到GPU
    print(f"\n强制移动模型到 {target_device}")
    encoder = encoder.to(target_device)
    predictor = predictor.to(target_device)
    
    print(f"移动后，encoder参数设备: {next(encoder.parameters()).device}")
    print(f"移动后，predictor参数设备: {next(predictor.parameters()).device}")
    
    print("\n=== 移动模型后的GPU状态 ===")
    check_gpu_usage()
    
    # 5. 创建测试张量
    print(f"\n=== 创建测试张量 ===")
    test_tensor = torch.randn(1000, 1000, device=target_device)
    print(f"测试张量设备: {test_tensor.device}")
    print(f"测试张量占用内存: {test_tensor.element_size() * test_tensor.nelement() / 1024**2:.2f} MB")
    
    print("\n=== 创建张量后的GPU状态 ===")
    check_gpu_usage()
    
    # 6. 运行简单前向传播测试
    print(f"\n=== 测试模型前向传播 ===")
    # 创建模拟输入
    batch_size = 1
    seq_len = 2
    height, width = 256, 256
    channels = 3
    
    # 模拟视频输入
    video_input = torch.randn(batch_size, seq_len, channels, height, width, device=target_device)
    print(f"视频输入设备: {video_input.device}, 形状: {video_input.shape}")
    
    # 重新整形为编码器期望的格式
    B, T, C, H, W = video_input.shape
    encoder_input = video_input.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    print(f"编码器输入设备: {encoder_input.device}, 形状: {encoder_input.shape}")
    
    print("\n=== 前向传播前的GPU状态 ===")
    check_gpu_usage()
    
    # 运行编码器
    with torch.no_grad():
        start_time = time.time()
        encoded = encoder(encoder_input)
        torch.cuda.synchronize()  # 确保GPU操作完成
        end_time = time.time()
    
    print(f"编码器输出设备: {encoded.device}, 形状: {encoded.shape}")
    print(f"编码器运行时间: {end_time - start_time:.3f}秒")
    
    print("\n=== 前向传播后的GPU状态 ===")
    check_gpu_usage()
    
    # 清理内存
    del test_tensor, video_input, encoder_input, encoded
    torch.cuda.empty_cache()
    
    print("\n=== 清理后的GPU状态 ===")
    check_gpu_usage()
    
    return encoder, predictor

if __name__ == "__main__":
    encoder, predictor = debug_model_placement()