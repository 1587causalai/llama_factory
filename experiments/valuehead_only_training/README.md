# ValueHead 单独训练实验

## 实验目的

本实验旨在研究 LEDPO 算法中的 ValueHead 网络单独训练的效果。通过冻结主干网络 (policy)，只训练 ValueHead 网络，观察：

1. 模型的精度变化 - 预期精度应该保持不变，因为生成策略没有变化
2. 训练损失变化 - 预期损失会下降，因为 ValueHead 在优化
3. 不同学习率倍率下 ValueHead 的训练表现

## 实验假设

1. ValueHead 网络在冻结主干网络的情况下能够学习有效的 beta 值
2. 更高的 ValueHead 学习率可能导致更快的收敛
3. 单独训练 ValueHead 和联合训练相比可能有不同的性能特点

## 使用方法

### 1. 运行单个倍率的实验

```bash
conda activate llama
python experiments/valuehead_only_training/run_valuehead_experiments.py --multipliers 100.0
```

### 2. 运行多个倍率的实验

```bash
conda activate llama
python experiments/valuehead_only_training/run_valuehead_experiments.py --multipliers 10.0 100.0 1000.0
```

### 3. 使用默认倍率集合

```bash
conda activate llama
python experiments/valuehead_only_training/run_valuehead_experiments.py
```

将自动运行 [10.0, 100.0, 1000.0, 5000.0] 这几个倍率的实验。

### 4. 指定 Wandb 项目

```bash
conda activate llama
python experiments/valuehead_only_training/run_valuehead_experiments.py --wandb-project my_project_name
```

## 实验配置

- 基础模型: Qwen1.5-0.5B
- 冻结设置: `freeze_policy=true`，冻结所有主干网络参数
- 训练数据: alpaca_zh 数据集
- 训练方式: LEDPO + LoRA
- 最大样本数: 500 (快速实验)

## 预期结果

1. 训练损失应该随着训练进行而下降
2. 评估准确度应该保持大致不变
3. ValueHead 学习率倍率越高，训练越快收敛
4. 较大的 ValueHead 学习率可能导致训练不稳定

## 后续研究方向

1. 将此方法扩展到更大的模型
2. 尝试两阶段训练：先训练 Policy，再训练 ValueHead
3. 比较冻结训练与联合训练的效果差异 