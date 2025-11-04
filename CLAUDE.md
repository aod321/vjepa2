# V-JEPA 2 项目概览

## 项目简介

V-JEPA 2 (Video Joint-Embedding Predictive Architecture 2) 是 Meta FAIR 开发的自监督视频表示学习框架，基于掩码预测任务学习视频表示。项目包含两个主要组件：

1. **V-JEPA 2**: 基础的自监督视频编码器
2. **V-JEPA 2-AC**: 动作条件世界模型，用于机器人控制任务

## 核心概念

### 1. 模型架构

#### 编码器 (Vision Transformer)
- 基于 ViT 架构，支持多种规模（base/large/huge/giant）
- 使用 3D patch embedding 处理视频输入
- 支持 RoPE（旋转位置编码）和传统位置编码
- 包含 EMA 目标编码器用于稳定训练

#### 预测器 (Predictor)
- **标准预测器**: 预测被掩码的视频表示
- **动作条件预测器 (AC)**: 接受动作/状态输入，支持世界模型任务
  - 额外的 action_encoder 和 state_encoder
  - 使用因果注意力掩码确保时序关系
  - 支持机器人控制等任务

### 2. 世界模型与控制

#### CEM (Cross-Entropy Method) 算法
位置：`notebooks/utils/mpc_utils.py`

CEM 是一种基于采样的优化算法，用于动作轨迹规划：

```python
核心流程：
1. 初始化动作分布 N(mean, std)
2. 迭代优化：
   - 从当前分布采样多条轨迹（如100条）
   - 用世界模型预测每条轨迹的结果
   - 选择最接近目标的 top-k 条轨迹
   - 用 top-k 轨迹更新分布参数（动量更新）
3. 返回优化后的 mean 作为最优动作
```

关键参数：
- `rollout`: 预测的时间步数
- `samples`: 每次迭代的采样数量
- `topk`: 选择的最优轨迹数
- `momentum_mean/std`: 更新时的动量系数

#### 动作表示
7维动作空间：`[x, y, z, rx, ry, rz, gripper]`
- 前3维：位置变化（通常限制在 [-maxnorm, maxnorm]）
- 中间3维：旋转变化（示例中通常设为0）
- 最后1维：夹爪开合（范围 [-0.75, 0.75]）

#### 轨迹预测
```python
# 初始状态
pose_traj = context_pose  # 机器人当前位姿
frame_traj = context_frame  # 当前视觉观察

# 循环预测多步
for h in range(rollout):
    # 采样动作
    action = sample_from_distribution(mean[h], std[h])
    # 预测下一状态
    next_frame, next_pose = world_model(frame_traj, action, pose_traj)
    # 更新轨迹
    frame_traj = concat(frame_traj, next_frame)
    pose_traj = concat(pose_traj, next_pose)
```

### 3. 数据流程

#### 训练时
1. 视频输入 → 数据增强 → 归一化
2. 生成多尺度时空掩码
3. 编码器提取特征（上下文和目标）
4. 预测器预测被掩码的表示
5. 计算 L2 损失

#### 推理时（世界模型控制）
1. 编码当前观察和目标观察
2. 使用 CEM 优化动作序列
3. 选择第一个动作执行
4. 重复规划（滚动时域控制）

## 项目结构

```
vjepa2/
├── app/                    # 训练脚本
│   ├── vjepa/             # 基础模型训练
│   └── vjepa_droid/       # 动作条件模型训练
├── src/
│   ├── models/            # 模型定义
│   │   ├── vision_transformer.py
│   │   ├── predictor.py
│   │   └── ac_predictor.py
│   ├── datasets/          # 数据加载
│   └── masks/             # 掩码生成
├── notebooks/             # 示例和工具
│   └── utils/
│       ├── mpc_utils.py   # CEM实现
│       └── world_model_wrapper.py
└── evals/                 # 评估脚本
```

## 关键代码位置

1. **CEM 算法实现**: `notebooks/utils/mpc_utils.py:28-163`
2. **动作条件预测器**: `src/models/ac_predictor.py`
3. **世界模型包装器**: `notebooks/utils/world_model_wrapper.py`
4. **示例脚本**: `worldmodel_control.py`

## 常见任务

### 1. 使用世界模型进行控制
```python
# 初始化模型
encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")

# 创建世界模型
world_model = WorldModel(
    encoder=encoder,
    predictor=predictor,
    mpc_args={
        "rollout": 2,
        "samples": 100,
        "topk": 10,
        "cem_steps": 100
    }
)

# 规划动作
action = world_model.infer_next_action(current_rep, current_pose, goal_rep)
```

### 2. 调整 CEM 参数
- 增加 `samples` 和 `cem_steps` 提高精度但增加计算时间
- 调整 `momentum_mean/std` 控制收敛速度
- 修改 `maxnorm` 限制动作幅度
- 使用 `close_gripper` 参数控制抓取时机

### 3. 处理约束
- 使用 `axis` 参数固定某些自由度
- 通过 `clip` 限制动作范围
- 添加自定义目标函数替代默认的 L1 距离

## 性能指标

模型在多个基准测试上达到 SOTA：
- Epic-Kitchens-100: 39.7%
- Something-Something v2: 77.3%
- Diving48: 90.2%

## 注意事项

1. **时序因果性**: 动作条件模型使用因果注意力，确保不会"看到"未来
2. **动作空间**: 当前实现主要支持笛卡尔空间控制
3. **计算资源**: CEM 优化可能需要较多计算，建议使用 GPU
4. **轨迹长度**: 更长的 rollout 提供更好的规划但计算成本更高

## 相关链接

- [论文](https://ai.meta.com/research/publications/video-joint-embedding-predictive-architecture-v-jepa/)
- [GitHub](https://github.com/facebookresearch/vjepa2)
- [模型权重](https://huggingface.co/facebook/vjepa2)