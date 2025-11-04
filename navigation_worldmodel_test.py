#%%
"""
Navigation World Model Analysis Script
导航世界模型分析脚本 - 测试go_stanford数据集上训练的模型效果

使用Jupyter cell风格 (#%%)，可在VS Code中逐步运行
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import pickle
import h5py
import cv2
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

# 导入V-JEPA相关模块
from app.vjepa_droid.transforms import make_transforms
from notebooks.utils.world_model_wrapper import WorldModel
from notebooks.utils.mpc_utils import compute_new_pose, poses_to_diff

# 设置随机种子以保证复现性
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print("✅ 导入完成，随机种子已设置")

#%%
"""
Cell 1: 加载训练好的导航模型
"""

# 模型路径和参数
# checkpoint_path = "/nvmessd/yinzi/vjepa2/checkpoints/go_stanford_finetune_8gpu_0818_12_18/e20.pt"
checkpoint_path = "/nvmessd/yinzi/vjepa2/checkpoints/go_stanford_finetune_8gpu/latest.pt"
crop_size = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔧 使用设备: {device}")
print(f"📂 加载checkpoint: {checkpoint_path}")

# 检查checkpoint文件是否存在
if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint文件不存在: {checkpoint_path}")
    print("请检查路径是否正确")
else:
    print("✅ Checkpoint文件存在")

# 加载预训练的V-JEPA2-AC模型作为基础
print("📥 加载基础V-JEPA2-AC模型...")
encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")

# 加载fine-tuned权重
if os.path.exists(checkpoint_path):
    print("📥 加载fine-tuned权重...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    def remove_module_prefix(state_dict):
        """移除state_dict中的'module.'前缀"""
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 移除'module.'前缀
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict
    
    # 加载encoder权重
    if 'encoder' in checkpoint:
        encoder_state_dict = remove_module_prefix(checkpoint['encoder'])
        encoder.load_state_dict(encoder_state_dict, strict=False)
        print("✅ Encoder权重加载成功")
    
    # 加载predictor权重  
    if 'predictor' in checkpoint:
        predictor_state_dict = remove_module_prefix(checkpoint['predictor'])
        predictor.load_state_dict(predictor_state_dict, strict=False)
        print("✅ Predictor权重加载成功")
    
    print(f"📊 训练轮数: {checkpoint.get('epoch', 'unknown')}")
else:
    print("⚠️ 使用预训练权重（未加载fine-tuned）")

# 移动到指定设备
encoder = encoder.to(device)
predictor = predictor.to(device)
encoder.eval()
predictor.eval()

print("✅ 模型加载完成")

#%%
"""
Cell 2: 初始化数据变换和模型参数
"""

# 计算tokens_per_frame
tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)
print(f"🎯 Tokens per frame: {tokens_per_frame}")

# 初始化数据变换
transform = make_transforms(
    random_horizontal_flip=False,
    random_resize_aspect_ratio=(1., 1.),
    random_resize_scale=(1., 1.),
    reprob=0.,
    auto_augment=False,
    motion_shift=False,
    crop_size=crop_size,
)

print("✅ 数据变换初始化完成")

#%%
"""
Cell 3: 导航数据集路径和选择测试样本
"""

# 导航数据集路径
csv_path = "/nvmessd/yinzi/vjepa2/go_stanford_converted/go_stanford_train_paths.csv"

print(f"📁 读取数据集索引: {csv_path}")

# 读取所有轨迹路径
if not os.path.exists(csv_path):
    print(f"❌ CSV文件不存在: {csv_path}")
    sys.exit(1)

with open(csv_path, 'r') as f:
    episode_paths = [line.strip() for line in f.readlines() if line.strip()]

print(f"📊 总共找到 {len(episode_paths)} 个导航轨迹")

# 检查轨迹长度分布
print("🔍 检查轨迹长度分布...")
trajectory_lengths = []
valid_trajectories = []

for i, ep_path in enumerate(episode_paths[:20]):  # 检查前20个
    try:
        ep_clips, ep_states = load_navigation_episode(ep_path)
        length = len(ep_states)
        trajectory_lengths.append(length)
        if length >= 10:  # 至少10帧的轨迹
            valid_trajectories.append(ep_path)
        if i < 10:
            print(f"   {os.path.basename(ep_path)}: {length} 帧")
    except:
        continue

if trajectory_lengths:
    print(f"📊 轨迹长度统计:")
    print(f"   平均长度: {np.mean(trajectory_lengths):.1f} 帧")
    print(f"   最短: {np.min(trajectory_lengths)} 帧")
    print(f"   最长: {np.max(trajectory_lengths)} 帧")
    print(f"   ≥10帧的轨迹数: {len(valid_trajectories)}")
    
    if len(valid_trajectories) > 0:
        print(f"✅ 找到 {len(valid_trajectories)} 个可用轨迹")
        # 使用找到的有效轨迹
        episode_paths = valid_trajectories
    else:
        print("❌ 没有找到足够长的轨迹")
else:
    print("❌ 无法读取轨迹信息")

# 随机选择一个测试轨迹（固定种子保证复现）
test_episode_path = random.choice(episode_paths)
print(f"🎯 选择测试轨迹: {os.path.basename(test_episode_path)}")

# 检查轨迹文件
video_path = os.path.join(test_episode_path, "recordings/MP4/nav_camera.mp4")
traj_path = os.path.join(test_episode_path, "trajectory.h5")

if not os.path.exists(video_path):
    print(f"❌ 视频文件不存在: {video_path}")
if not os.path.exists(traj_path):
    print(f"❌ 轨迹文件不存在: {traj_path}")

print("✅ 测试轨迹选择完成")

#%%
"""
Cell 4: 加载和预处理测试数据
"""

def load_navigation_episode(episode_path):
    """加载导航轨迹数据"""
    video_path = os.path.join(episode_path, "recordings/MP4/nav_camera.mp4")
    traj_path = os.path.join(episode_path, "trajectory.h5")
    
    # 加载视频
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换BGR到RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    # 转换为numpy数组
    frames = np.array(frames)  # [T, H, W, C]
    frames = frames.transpose(0, 3, 1, 2)  # [T, C, H, W]
    
    # 加载轨迹数据
    with h5py.File(traj_path, 'r') as f:
        cartesian_positions = f['observation/robot_state/cartesian_position'][:]
        gripper_positions = f['observation/robot_state/gripper_position'][:]
    
    # 合并状态 [x, y, z, rx, ry, rz, gripper]
    states = np.concatenate([cartesian_positions, gripper_positions[:, None]], axis=1)
    
    return frames, states

print("📥 加载测试轨迹数据...")
np_clips, np_states = load_navigation_episode(test_episode_path)

# 检查数据维度
print(f"📊 视频维度: {np_clips.shape}")  # [T, C, H, W]
print(f"📊 状态维度: {np_states.shape}")  # [T, 7]
print(f"📊 轨迹长度: {len(np_states)} 步")

# 显示位置和角度范围
pos_range_x = (np_states[:, 0].min(), np_states[:, 0].max())
pos_range_y = (np_states[:, 1].min(), np_states[:, 1].max())
yaw_range = (np_states[:, 5].min(), np_states[:, 5].max())

print(f"📍 X位置范围: {pos_range_x[0]:.2f} ~ {pos_range_x[1]:.2f} m")
print(f"📍 Y位置范围: {pos_range_y[0]:.2f} ~ {pos_range_y[1]:.2f} m") 
print(f"🧭 Yaw角度范围: {yaw_range[0]:.2f} ~ {yaw_range[1]:.2f} rad ({np.degrees(yaw_range[0]):.1f}° ~ {np.degrees(yaw_range[1]):.1f}°)")

# 转换为torch tensors
# np_clips shape: [T, C, H, W] -> need [T, H, W, C] for transform
np_clips_for_transform = np_clips.transpose(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]
clips = transform(np_clips_for_transform).unsqueeze(0).to(device)  # [1, C, T, H, W]
states = torch.tensor(np_states, dtype=torch.float32, device=device)

print("✅ 测试数据加载完成")

#%%
"""
Cell 5: 可视化原始轨迹数据
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 左图：2D轨迹
ax1.plot(np_states[:, 0], np_states[:, 1], 'b-', linewidth=2, alpha=0.8, label='Navigation Path')
ax1.plot(np_states[0, 0], np_states[0, 1], 'go', markersize=10, label='Start', zorder=5)
ax1.plot(np_states[-1, 0], np_states[-1, 1], 'ro', markersize=10, label='Goal', zorder=5)

# 添加方向箭头
arrow_interval = max(1, len(np_states) // 10)
for i in range(0, len(np_states), arrow_interval):
    dx = 0.3 * np.cos(np_states[i, 5])
    dy = 0.3 * np.sin(np_states[i, 5])
    ax1.arrow(np_states[i, 0], np_states[i, 1], dx, dy, 
             head_width=0.1, head_length=0.05, fc='blue', alpha=0.6)

ax1.set_xlabel('X (meters)')
ax1.set_ylabel('Y (meters)')
ax1.set_title('Original Navigation Trajectory')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# 右图：视频帧拼接显示
T = min(10, len(np_clips))  # 显示前10帧
frame_mosaic = np.concatenate([np_clips[i].transpose(1, 2, 0) for i in range(T)], axis=1)
ax2.imshow(frame_mosaic)
ax2.set_title(f'First {T} Video Frames')
ax2.axis('off')

plt.tight_layout()
plt.show()

print("✅ 原始数据可视化完成")

#%%
"""
Cell 6: 设置测试场景参数
"""

# 测试场景设置
trajectory_length = len(states)
start_idx = max(0, trajectory_length // 2 - 2)  # 从中间偏前开始
prediction_horizon = 3                           # 预测3步（更现实的短期预测）
goal_idx = min(start_idx + prediction_horizon, trajectory_length - 1)  # 目标是3步后
context_length = 2                               # 使用2帧作为上下文

print(f"🎯 测试场景设置:")
print(f"   轨迹总长度: {trajectory_length} 步")
print(f"   起始位置: 第 {start_idx} 步")  
print(f"   目标位置: 第 {goal_idx} 步")
print(f"   预测长度: {goal_idx - start_idx} 步 (短期预测)")
print(f"   上下文长度: {context_length} 帧")

# 提取测试片段
test_clips = clips[:, :, start_idx:start_idx+context_length]  # [1, C, context_length, H, W]
test_states = states[start_idx:start_idx+context_length]      # [context_length, 7]
goal_clips = clips[:, :, goal_idx:goal_idx+1]                # [1, C, 1, H, W]
goal_states = states[goal_idx:goal_idx+1]                     # [1, 7]

# 显示当前位置和目标位置
current_pos = test_states[-1, [0, 1]].cpu().numpy()
goal_pos = goal_states[0, [0, 1]].cpu().numpy()
distance_to_goal = np.linalg.norm(goal_pos - current_pos)

print(f"📍 当前位置: ({current_pos[0]:.2f}, {current_pos[1]:.2f})")
print(f"📍 目标位置: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
print(f"📏 直线距离: {distance_to_goal:.2f} m")

print("✅ 测试场景参数设置完成")

#%%
"""
Cell 7: 设置导航专用的世界模型
"""

def make_navigation_action_grid(grid_size_xy=0.1, grid_size_yaw=0.2, nsamples=3, device='cpu'):
    """
    导航专用动作采样：只在 [x, y, yaw] 维度采样
    返回 nsamples^3 个动作样本
    """
    action_samples = []
    for dx in np.linspace(-grid_size_xy, grid_size_xy, nsamples):
        for dy in np.linspace(-grid_size_xy, grid_size_xy, nsamples):
            for dyaw in np.linspace(-grid_size_yaw, grid_size_yaw, nsamples):
                # 构造7DOF动作：[x, y, z=0, rx=0, ry=0, rz=yaw, gripper=0]
                action = torch.tensor([dx, dy, 0.0, 0.0, 0.0, dyaw, 0.0], 
                                    device=device, dtype=torch.float32)
                action_samples.append(action)
    return torch.stack(action_samples, dim=0).unsqueeze(1)  # [N, 1, 7]

# 导航专用的世界模型配置
navigation_mpc_args = {
    "rollout": 2,                # 预测步数
    "samples": 27,               # 3^3 = 27个动作样本
    "topk": 9,                   # 选择top-9
    "cem_steps": 3,              # CEM迭代次数
    "momentum_mean": 0.2,        # 均值动量
    "momentum_std": 0.6,         # 标准差动量  
    "maxnorm": 0.15,            # 最大动作幅度
    "verbose": True              # 输出详细信息
}

print(f"🤖 世界模型配置:")
for key, value in navigation_mpc_args.items():
    print(f"   {key}: {value}")

# 创建世界模型包装器
world_model = WorldModel(
    encoder=encoder,
    predictor=predictor,
    tokens_per_frame=tokens_per_frame,
    transform=lambda x: x,  # 数据已经预处理过了
    mpc_args=navigation_mpc_args,
    normalize_reps=True,
    device=device
)

print("✅ 导航世界模型初始化完成")

#%%
"""  
Cell 8: 运行世界模型预测
"""

print("🚀 开始世界模型预测...")

# 前向传播获取表示
def forward_target(clips_batch, normalize_reps=True):
    """获取视频帧的编码表示"""
    B, C, T, H, W = clips_batch.size()
    clips_reshaped = clips_batch.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    
    with torch.no_grad():
        h = encoder(clips_reshaped)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        if normalize_reps:
            h = torch.nn.functional.layer_norm(h, (h.size(-1),))
    return h

# 获取上下文和目标的表示
print("🔮 编码上下文帧...")
context_rep = forward_target(test_clips)  # [1, context_length * tokens_per_frame, D]

print("🎯 编码目标帧...")  
goal_rep = forward_target(goal_clips)     # [1, tokens_per_frame, D]

print("🧠 运行CEM优化...")
with torch.no_grad():
    # 提取上下文表示和状态
    z_context = context_rep[:, :tokens_per_frame]  # 最新帧的表示
    z_goal = goal_rep[:, -tokens_per_frame:]       # 目标帧的表示
    s_context = test_states[-1:].unsqueeze(0)     # [1, 1, 7] 当前状态
    
    # 使用世界模型规划下一步动作
    planned_actions = world_model.infer_next_action(z_context, s_context, z_goal)
    
print(f"✅ 规划完成！")
print(f"🎮 规划的动作: {planned_actions[0].cpu().numpy()}")

# 将动作分解显示
action_np = planned_actions[0].cpu().numpy()
print(f"   位移 (x, y): ({action_np[0]:.3f}, {action_np[1]:.3f}) m")
print(f"   旋转 (yaw): {action_np[5]:.3f} rad ({np.degrees(action_np[5]):.1f}°)")

#%%
"""
Cell 9: 模拟执行预测动作并可视化
"""

def simulate_action_execution(current_state, action):
    """模拟执行一个动作，返回新状态"""
    new_state = current_state.clone()
    new_state[0] += action[0]  # x
    new_state[1] += action[1]  # y  
    new_state[5] += action[5]  # yaw
    return new_state

# 从当前位置开始，执行多步预测
print("🔄 模拟多步预测执行...")

predicted_states = [test_states[-1]]  # 起始状态
current_state = test_states[-1].clone()

# 简单策略：重复执行相同的动作（实际应该每步重新规划）
n_prediction_steps = goal_idx - start_idx  # 预测到目标位置

for step in range(n_prediction_steps):
    print(f"   步骤 {step+1}/{n_prediction_steps}")
    
    # 执行动作
    new_state = simulate_action_execution(current_state, planned_actions[0])
    predicted_states.append(new_state)
    current_state = new_state

# 转换为numpy进行可视化
predicted_states_np = torch.stack(predicted_states).cpu().numpy()

# 真实轨迹对比数据
true_states_segment = states[start_idx:start_idx+len(predicted_states)].cpu().numpy()

print("✅ 预测执行完成")

#%%
"""
Cell 10: 结果可视化和评估
"""

def normalize_angle(angle):
    """将角度规范化到[-pi, pi]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def compute_angle_diff(angle1, angle2):
    """计算角度差"""
    diff = angle2 - angle1
    return normalize_angle(diff)

# 计算评估指标
position_errors = []
angle_errors = []

for i in range(len(true_states_segment)):
    # 位置误差
    pos_error = np.linalg.norm(
        true_states_segment[i, [0, 1]] - predicted_states_np[i, [0, 1]]
    )
    position_errors.append(pos_error)
    
    # 角度误差
    angle_error = abs(compute_angle_diff(
        true_states_segment[i, 5], predicted_states_np[i, 5]
    ))
    angle_errors.append(angle_error)

# 可视化结果
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 第1子图：2D轨迹对比
ax1.plot(true_states_segment[:, 0], true_states_segment[:, 1], 
         'b-', linewidth=3, label='True Path', alpha=0.8)
ax1.plot(predicted_states_np[:, 0], predicted_states_np[:, 1], 
         'r--', linewidth=3, label='Predicted Path', alpha=0.8)

# 标记起点和终点
ax1.plot(true_states_segment[0, 0], true_states_segment[0, 1], 
         'go', markersize=12, label='Start', zorder=5)
ax1.plot(goal_pos[0], goal_pos[1], 
         'mo', markersize=12, label='Target Goal', zorder=5)

# 添加方向箭头
for i in range(0, len(true_states_segment), max(1, len(true_states_segment)//3)):
    # 真实方向
    dx = 0.2 * np.cos(true_states_segment[i, 5])
    dy = 0.2 * np.sin(true_states_segment[i, 5])
    ax1.arrow(true_states_segment[i, 0], true_states_segment[i, 1], dx, dy,
             head_width=0.05, head_length=0.03, fc='blue', alpha=0.7)
    
    # 预测方向
    dx = 0.2 * np.cos(predicted_states_np[i, 5])
    dy = 0.2 * np.sin(predicted_states_np[i, 5])
    ax1.arrow(predicted_states_np[i, 0], predicted_states_np[i, 1], dx, dy,
             head_width=0.05, head_length=0.03, fc='red', alpha=0.7)

ax1.set_xlabel('X (meters)')
ax1.set_ylabel('Y (meters)')
ax1.set_title('Navigation World Model Prediction Results')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# 第2子图：位置误差
ax2.plot(range(len(position_errors)), position_errors, 'g-o', linewidth=2, markersize=6)
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Success Threshold (0.5m)')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Position Error (m)')
ax2.set_title('Position Error Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 第3子图：角度误差
ax3.plot(range(len(angle_errors)), np.degrees(angle_errors), 'orange', linewidth=2, marker='s', markersize=6)
ax3.axhline(y=np.degrees(0.2), color='red', linestyle='--', alpha=0.7, label='Success Threshold (11.5°)')
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Angle Error (degrees)')
ax3.set_title('Angle Error Over Time') 
ax3.legend()
ax3.grid(True, alpha=0.3)

# 第4子图：统计摘要
stats_text = f"""Prediction Results Summary:

📊 Trajectory Info:
   - Prediction steps: {len(predicted_states_np)}
   - Distance to goal: {distance_to_goal:.2f} m

📍 Position Metrics:
   - Mean error: {np.mean(position_errors):.3f} m
   - Final error: {position_errors[-1]:.3f} m  
   - Max error: {np.max(position_errors):.3f} m
   - Success rate: {np.mean(np.array(position_errors) < 0.5):.1%}

🧭 Angle Metrics:
   - Mean error: {np.degrees(np.mean(angle_errors)):.1f}°
   - Final error: {np.degrees(angle_errors[-1]):.1f}°
   - Max error: {np.degrees(np.max(angle_errors)):.1f}°
   - Success rate: {np.mean(np.array(angle_errors) < 0.2):.1%}

🎯 Overall Performance:
   - Combined success: {np.mean((np.array(position_errors) < 0.5) & (np.array(angle_errors) < 0.2)):.1%}
"""

ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
         verticalalignment='top', fontfamily='monospace', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
ax4.axis('off')

plt.tight_layout()
plt.savefig('navigation_worldmodel_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 结果可视化完成")
print("💾 结果已保存为 'navigation_worldmodel_results.png'")

#%%
"""
Cell 11: 总结和结论
"""

print("\n" + "="*60)
print("🏁 导航世界模型测试总结")
print("="*60)

print(f"\n📁 测试配置:")
print(f"   模型checkpoint: {os.path.basename(checkpoint_path)}")  
print(f"   测试轨迹: {os.path.basename(test_episode_path)}")
print(f"   预测步数: {len(predicted_states_np)}")
print(f"   随机种子: {RANDOM_SEED}")

print(f"\n📊 性能指标:")
print(f"   平均位置误差: {np.mean(position_errors):.3f} m")
print(f"   平均角度误差: {np.degrees(np.mean(angle_errors)):.1f}°")
print(f"   位置成功率: {np.mean(np.array(position_errors) < 0.5):.1%} (< 0.5m)")
print(f"   角度成功率: {np.mean(np.array(angle_errors) < 0.2):.1%} (< 11.5°)")

print(f"\n🎯 关键发现:")
final_pos_error = position_errors[-1]
final_angle_error = angle_errors[-1]

if final_pos_error < 0.5:
    print("   ✅ 最终位置误差在可接受范围内")
else:
    print("   ❌ 最终位置误差较大，需要改进")

if final_angle_error < 0.2:
    print("   ✅ 最终角度误差在可接受范围内")
else:
    print("   ❌ 最终角度误差较大，需要改进")

print(f"\n💡 建议:")
if np.mean(position_errors) > 0.3:
    print("   - 考虑增加CEM采样数量或迭代次数")
    print("   - 调整动作采样范围")
if np.mean(angle_errors) > 0.15:
    print("   - 角度预测可能需要更精细的采样")
    print("   - 检查训练数据中的角度分布")

print("="*60)
print("🎉 测试完成！")

# %%
