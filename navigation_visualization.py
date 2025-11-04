"""
Navigation Control Visualization Tools
导航控制可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import torch

def normalize_angle(angle):
    """将角度规范化到[-pi, pi]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def compute_angle_diff(angle1, angle2):
    """计算两个角度之间的最短角距离"""
    diff = angle2 - angle1
    return normalize_angle(diff)

def visualize_navigation_trajectory(true_states, pred_states, current_step=None, 
                                  save_path=None, title_suffix="", show_arrows=True, 
                                  arrow_interval=None, figsize=(16, 7)):
    """
    2D鸟瞰图显示真实轨迹vs预测轨迹
    
    Args:
        true_states: 真实状态序列 [T, 7] - [x, y, z, rx, ry, rz, gripper]
        pred_states: 预测状态序列 [T, 7]
        current_step: 当前步数，用于高亮显示
        save_path: 保存路径
        title_suffix: 标题后缀
        show_arrows: 是否显示朝向箭头
        arrow_interval: 箭头显示间隔，None则自动计算
        figsize: 图像大小
    
    Returns:
        fig: matplotlib figure对象
    """
    
    # 确保输入是numpy数组
    if torch.is_tensor(true_states):
        true_states = true_states.cpu().numpy()
    if torch.is_tensor(pred_states):
        pred_states = pred_states.cpu().numpy()
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 提取位置和角度
    true_pos = true_states[:, [0, 1]]  # [x, y]
    true_yaw = true_states[:, 5]       # yaw角度
    pred_pos = pred_states[:, [0, 1]]
    pred_yaw = pred_states[:, 5]
    
    # === 左图：轨迹对比 ===
    
    # 绘制完整轨迹路径
    ax1.plot(true_pos[:, 0], true_pos[:, 1], 'b-', linewidth=3, 
             label='Ground Truth', alpha=0.8, zorder=2)
    ax1.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', linewidth=3, 
             label='Predicted', alpha=0.8, zorder=2)
    
    # 标记起点和终点
    ax1.plot(true_pos[0, 0], true_pos[0, 1], 'go', markersize=12, 
             label='Start', zorder=5, markeredgecolor='darkgreen', markeredgewidth=2)
    ax1.plot(true_pos[-1, 0], true_pos[-1, 1], 'mo', markersize=12, 
             label='Goal', zorder=5, markeredgecolor='purple', markeredgewidth=2)
    
    # 标记当前位置（如果指定）
    if current_step is not None and current_step < len(true_pos):
        ax1.plot(true_pos[current_step, 0], true_pos[current_step, 1], 
                'ko', markersize=10, label='Current', zorder=6, 
                markeredgecolor='white', markeredgewidth=2)
    
    # 绘制朝向箭头
    if show_arrows:
        # 自动计算箭头间隔
        if arrow_interval is None:
            arrow_interval = max(1, len(true_pos) // 8)  # 大约显示8个箭头
        
        arrow_length = 0.3  # 箭头长度
        
        for i in range(0, len(true_pos), arrow_interval):
            # 真实轨迹的朝向箭头
            dx = arrow_length * np.cos(true_yaw[i])
            dy = arrow_length * np.sin(true_yaw[i])
            
            arrow = FancyArrowPatch(
                (true_pos[i, 0], true_pos[i, 1]),
                (true_pos[i, 0] + dx, true_pos[i, 1] + dy),
                arrowstyle='->', mutation_scale=15, 
                color='blue', alpha=0.7, linewidth=2, zorder=3
            )
            ax1.add_patch(arrow)
            
            # 预测轨迹的朝向箭头
            if i < len(pred_pos):
                dx = arrow_length * np.cos(pred_yaw[i])
                dy = arrow_length * np.sin(pred_yaw[i])
                
                arrow = FancyArrowPatch(
                    (pred_pos[i, 0], pred_pos[i, 1]),
                    (pred_pos[i, 0] + dx, pred_pos[i, 1] + dy),
                    arrowstyle='->', mutation_scale=15, 
                    color='red', alpha=0.7, linewidth=2, zorder=3
                )
                ax1.add_patch(arrow)
    
    # 添加时间步标记（每隔几步显示数字）
    step_interval = max(1, len(true_pos) // 6)
    for i in range(0, len(true_pos), step_interval):
        ax1.annotate(f'{i}', (true_pos[i, 0], true_pos[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7, color='blue',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    # 设置轴标签和标题
    ax1.set_xlabel('X (meters)', fontsize=12)
    ax1.set_ylabel('Y (meters)', fontsize=12)
    ax1.set_title(f'Navigation Trajectory Comparison{title_suffix}', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # === 右图：误差随时间变化 ===
    
    time_steps = np.arange(len(true_pos))
    
    # 计算位置误差
    position_errors = np.linalg.norm(true_pos - pred_pos, axis=1)
    
    # 计算角度误差（考虑周期性）
    angle_errors = np.abs(np.array([compute_angle_diff(t, p) for t, p in zip(true_yaw, pred_yaw)]))
    
    # 创建双y轴
    ax2_twin = ax2.twinx()
    
    # 绘制误差曲线
    line1 = ax2.plot(time_steps, position_errors, 'g-', linewidth=2, 
                     marker='o', markersize=4, label='Position Error', alpha=0.8)
    line2 = ax2_twin.plot(time_steps, np.degrees(angle_errors), 'orange', 
                         linewidth=2, marker='s', markersize=4, 
                         label='Angle Error', alpha=0.8)
    
    # 添加误差阈值线
    ax2.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, 
                label='Pos Success Threshold (0.5m)')
    ax2_twin.axhline(y=np.degrees(0.2), color='orange', linestyle='--', alpha=0.5, 
                     label='Angle Success Threshold (11.5°)')
    
    # 高亮当前步数
    if current_step is not None and current_step < len(time_steps):
        ax2.axvline(x=current_step, color='red', linestyle=':', alpha=0.7, linewidth=2)
        ax2.plot(current_step, position_errors[current_step], 'ro', markersize=8, zorder=5)
        ax2_twin.plot(current_step, np.degrees(angle_errors[current_step]), 'ro', markersize=8, zorder=5)
    
    # 设置轴标签和颜色
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Position Error (m)', color='g', fontsize=12)
    ax2_twin.set_ylabel('Angle Error (degrees)', color='orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='g')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    
    # 设置标题
    ax2.set_title('Prediction Errors Over Time', fontsize=14)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', fontsize=10)
    
    # 添加网格
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息文本
    mean_pos_error = np.mean(position_errors)
    mean_angle_error = np.mean(angle_errors)
    final_pos_error = position_errors[-1]
    final_angle_error = angle_errors[-1]
    
    pos_success_rate = np.mean(position_errors < 0.5)
    angle_success_rate = np.mean(angle_errors < 0.2)
    
    stats_text = f"""Statistics:
Mean Pos Error: {mean_pos_error:.3f}m
Mean Angle Error: {np.degrees(mean_angle_error):.1f}°
Final Pos Error: {final_pos_error:.3f}m
Final Angle Error: {np.degrees(final_angle_error):.1f}°
Pos Success Rate: {pos_success_rate:.1%}
Angle Success Rate: {angle_success_rate:.1%}"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def compute_navigation_metrics(true_states, pred_states):
    """
    计算导航任务评估指标
    
    Args:
        true_states: 真实状态 [T, 7]
        pred_states: 预测状态 [T, 7]
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    # 确保输入是numpy数组
    if torch.is_tensor(true_states):
        true_states = true_states.cpu().numpy()
    if torch.is_tensor(pred_states):
        pred_states = pred_states.cpu().numpy()
    
    # 提取位置 [x, y]
    true_pos = true_states[:, [0, 1]]
    pred_pos = pred_states[:, [0, 1]]
    
    # 提取yaw角度
    true_yaw = true_states[:, 5]
    pred_yaw = pred_states[:, 5]
    
    # 1. 位置误差 (欧几里得距离)
    position_errors = np.linalg.norm(true_pos - pred_pos, axis=1)
    mean_pos_error = np.mean(position_errors)
    final_pos_error = position_errors[-1]
    max_pos_error = np.max(position_errors)
    
    # 2. 角度误差 (考虑周期性)
    angle_errors = np.abs(np.array([compute_angle_diff(t, p) for t, p in zip(true_yaw, pred_yaw)]))
    mean_angle_error = np.mean(angle_errors)
    final_angle_error = angle_errors[-1]
    max_angle_error = np.max(angle_errors)
    
    # 3. 成功率指标 (基于阈值)
    pos_success_rate = np.mean(position_errors < 0.5)    # 50cm内算成功
    angle_success_rate = np.mean(angle_errors < 0.2)     # ~11度内算成功
    combined_success_rate = np.mean((position_errors < 0.5) & (angle_errors < 0.2))
    
    # 4. 轨迹相似度 (使用Dynamic Time Warping的简化版本)
    trajectory_similarity = np.exp(-mean_pos_error)  # 简化的相似度度量
    
    # 5. 路径长度比较
    true_path_length = np.sum(np.linalg.norm(np.diff(true_pos, axis=0), axis=1))
    pred_path_length = np.sum(np.linalg.norm(np.diff(pred_pos, axis=0), axis=1))
    path_length_ratio = pred_path_length / true_path_length if true_path_length > 0 else 1.0
    
    return {
        # 位置指标
        'mean_position_error_m': mean_pos_error,
        'final_position_error_m': final_pos_error,
        'max_position_error_m': max_pos_error,
        
        # 角度指标
        'mean_angle_error_rad': mean_angle_error,
        'mean_angle_error_deg': np.degrees(mean_angle_error),
        'final_angle_error_rad': final_angle_error,
        'final_angle_error_deg': np.degrees(final_angle_error),
        'max_angle_error_rad': max_angle_error,
        'max_angle_error_deg': np.degrees(max_angle_error),
        
        # 成功率指标
        'position_success_rate': pos_success_rate,
        'angle_success_rate': angle_success_rate,
        'combined_success_rate': combined_success_rate,
        
        # 轨迹指标
        'trajectory_similarity': trajectory_similarity,
        'path_length_ratio': path_length_ratio,
        'true_path_length_m': true_path_length,
        'pred_path_length_m': pred_path_length,
        
        # 原始数据（用于进一步分析）
        'position_errors': position_errors,
        'angle_errors': angle_errors,
    }


def print_navigation_metrics(metrics, title="Navigation Metrics"):
    """
    打印格式化的导航评估指标
    
    Args:
        metrics: compute_navigation_metrics返回的指标字典
        title: 打印标题
    """
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50)
    
    print(f"\n📍 Position Metrics:")
    print(f"   Mean Error:     {metrics['mean_position_error_m']:.3f} m")
    print(f"   Final Error:    {metrics['final_position_error_m']:.3f} m")
    print(f"   Max Error:      {metrics['max_position_error_m']:.3f} m")
    print(f"   Success Rate:   {metrics['position_success_rate']:.1%} (< 0.5m)")
    
    print(f"\n🧭 Orientation Metrics:")
    print(f"   Mean Error:     {metrics['mean_angle_error_deg']:.1f}° ({metrics['mean_angle_error_rad']:.3f} rad)")
    print(f"   Final Error:    {metrics['final_angle_error_deg']:.1f}° ({metrics['final_angle_error_rad']:.3f} rad)")
    print(f"   Max Error:      {metrics['max_angle_error_deg']:.1f}° ({metrics['max_angle_error_rad']:.3f} rad)")
    print(f"   Success Rate:   {metrics['angle_success_rate']:.1%} (< 11.5°)")
    
    print(f"\n🎯 Overall Performance:")
    print(f"   Combined Success: {metrics['combined_success_rate']:.1%}")
    print(f"   Trajectory Similarity: {metrics['trajectory_similarity']:.3f}")
    print(f"   Path Length Ratio: {metrics['path_length_ratio']:.2f}")
    
    print(f"\n📏 Path Lengths:")
    print(f"   True Path:      {metrics['true_path_length_m']:.2f} m")
    print(f"   Predicted Path: {metrics['pred_path_length_m']:.2f} m")
    
    print("="*50 + "\n")


# 示例使用函数
def demo_visualization():
    """
    演示可视化功能的示例代码
    """
    # 生成示例数据
    T = 20
    t = np.linspace(0, 4*np.pi, T)
    
    # 真实轨迹：螺旋形路径
    true_states = np.zeros((T, 7))
    true_states[:, 0] = t * np.cos(t) * 0.2  # x
    true_states[:, 1] = t * np.sin(t) * 0.2  # y
    true_states[:, 5] = t + np.pi/2           # yaw
    
    # 预测轨迹：添加一些噪声
    pred_states = true_states.copy()
    pred_states[:, 0] += np.random.normal(0, 0.1, T)  # x噪声
    pred_states[:, 1] += np.random.normal(0, 0.1, T)  # y噪声
    pred_states[:, 5] += np.random.normal(0, 0.2, T)  # yaw噪声
    
    # 创建可视化
    fig = visualize_navigation_trajectory(
        true_states, pred_states, 
        current_step=10,
        title_suffix=" (Demo)",
        save_path="demo_navigation_viz.png"
    )
    
    # 计算和打印指标
    metrics = compute_navigation_metrics(true_states, pred_states)
    print_navigation_metrics(metrics, "Demo Navigation Results")
    
    plt.show()
    
    return fig, metrics


if __name__ == "__main__":
    # 运行演示
    demo_visualization()