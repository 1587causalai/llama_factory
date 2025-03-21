# DPO vs LEDPO 完整对比实验

本实验旨在全面对比标准DPO和可学习Beta的DPO (LEDPO) 在大规模偏好数据上的效果差异。

## 实验背景

DPO (Direct Preference Optimization) 是一种基于人类偏好数据直接优化模型的方法。标准DPO使用固定的beta值（通常为0.1）作为温度参数，而LEDPO (Learnable Beta DPO) 则允许模型自动学习每个训练样本的最优beta值，理论上能提高模型对人类偏好的学习效果。

相比于简单的demo实验，本实验使用更完整的数据集、更长的训练时间和更严格的评估方法，以全面评估LEDPO的实际效果。

## 实验目标

1. 对比标准DPO和LEDPO在大规模训练数据上的性能差异
2. 分析LEDPO中beta值的学习过程和分布特征
3. 研究不同类型训练样本对beta值收敛的影响
4. 评估动态beta对模型偏好对齐的提升效果
5. 对比不同基础模型（Qwen1.5-0.5B vs Qwen2.5-1.5B-Instruct）对结果的影响

## 实验配置

### 实验一：Qwen1.5-0.5B

- **基础模型**: Qwen1.5-0.5B
- **微调方法**: LoRA (rank=8)
- **训练数据集**: hh_rlhf_en (Anthropic的Helpful-Harmless数据集)
- **训练参数**:
  - 批次大小: 2
  - 梯度累积步数: 4
  - 学习率: 1.0e-4
  - 训练轮次: 2.0
  - 优化器: AdamW
  - 学习率调度: cosine

### 实验二：Qwen2.5-1.5B-Instruct

- **基础模型**: Qwen2.5-1.5B-Instruct
- **微调方法**: LoRA (rank=8)
- **训练数据集**: hh_rlhf_en (Anthropic的Helpful-Harmless数据集)
- **训练参数**:
  - 批次大小: 2
  - 梯度累积步数: 4
  - 学习率: 1.0e-4
  - 训练轮次: 2.0
  - 优化器: AdamW
  - 学习率调度: cosine

## 实验方案

每个实验分为两组:

1. **标准DPO** - 使用固定的beta值(0.1)
2. **LEDPO** - 使用可学习的动态beta值，初始值设为0.1

每组实验将进行以下评估:

1. 训练过程监控 - 损失、奖励、准确率等指标
2. 模型质量评估 - 在评估集上的性能表现
3. beta值分析 - 分析LEDPO中beta值的变化特性（仅LEDPO）

## 运行实验

### 一键运行所有实验 (推荐)

我们提供了一个一键运行脚本，可以同时执行所有4个实验（Qwen1.5-DPO、Qwen1.5-LEDPO、Qwen2.5-DPO、Qwen2.5-LEDPO），并使用统一的Wandb项目名称 `ledpo_fullscale_experiment` 进行跟踪：

```bash
bash experiments/dpo_vs_ledpo_full/run_all_experiments.sh
```

该脚本支持以下参数：
- `--skip-qwen15`: 跳过所有Qwen1.5-0.5B模型实验
- `--skip-qwen25`: 跳过所有Qwen2.5-1.5B-Instruct模型实验
- `--skip-dpo`: 跳过所有标准DPO实验
- `--skip-ledpo`: 跳过所有LEDPO实验

例如，只运行Qwen2.5模型的实验：
```bash
bash experiments/dpo_vs_ledpo_full/run_all_experiments.sh --skip-qwen15
```

### 单独运行各组实验

如果需要分别运行各组实验，也可以使用以下命令：

#### 实验一：Qwen1.5-0.5B

```bash
bash experiments/dpo_vs_ledpo_full/run_experiment.sh
```

跳过部分实验:

```bash
# 只运行标准DPO
bash experiments/dpo_vs_ledpo_full/run_experiment.sh --skip-ledpo

# 只运行LEDPO
bash experiments/dpo_vs_ledpo_full/run_experiment.sh --skip-dpo
```

#### 实验二：Qwen2.5-1.5B-Instruct

```bash
bash experiments/dpo_vs_ledpo_full/run_qwen25_experiment.sh
```

跳过部分实验:

```bash
# 只运行标准DPO
bash experiments/dpo_vs_ledpo_full/run_qwen25_experiment.sh --skip-ledpo

# 只运行LEDPO
bash experiments/dpo_vs_ledpo_full/run_qwen25_experiment.sh --skip-dpo
```

## 结果分析

运行分析脚本:

```bash
# 分析Qwen1.5-0.5B实验结果
python experiments/dpo_vs_ledpo_full/analyze_logs.py \
    --dpo_dir experiments/dpo_vs_ledpo_full/results/dpo \
    --ledpo_dir experiments/dpo_vs_ledpo_full/results/ledpo \
    --output_dir experiments/dpo_vs_ledpo_full/analysis

# 分析Qwen2.5-1.5B-Instruct实验结果
python experiments/dpo_vs_ledpo_full/analyze_logs.py \
    --dpo_dir experiments/dpo_vs_ledpo_full/results/dpo_qwen25 \
    --ledpo_dir experiments/dpo_vs_ledpo_full/results/ledpo_qwen25 \
    --output_dir experiments/dpo_vs_ledpo_full/analysis_qwen25
```

分析脚本将生成详细的实验报告，包括:

1. 训练曲线对比图
2. 评估指标对比表格
3. Beta值分布和变化趋势分析
4. 统计显著性检验结果
5. 针对不同类型问题的性能差异分析

## 结果位置

### 实验一：Qwen1.5-0.5B
- **标准DPO结果**: `experiments/dpo_vs_ledpo_full/results/dpo`
- **LEDPO结果**: `experiments/dpo_vs_ledpo_full/results/ledpo`
- **分析结果**: `experiments/dpo_vs_ledpo_full/analysis`

### 实验二：Qwen2.5-1.5B-Instruct
- **标准DPO结果**: `experiments/dpo_vs_ledpo_full/results/dpo_qwen25`
- **LEDPO结果**: `experiments/dpo_vs_ledpo_full/results/ledpo_qwen25`
- **分析结果**: `experiments/dpo_vs_ledpo_full/analysis_qwen25`

## 预期结果

如果LEDPO的动态beta确实有助于提升模型性能，我们预期:

1. LEDPO在训练后期能达到比标准DPO更低的损失和更高的准确率
2. LEDPO对不同类型的问题会学习到不同的beta值，表明动态beta能适应不同的训练样本特性
3. 更大的模型（Qwen2.5-1.5B-Instruct）可能会从动态beta中获得更多或不同的收益

## 实验跟踪

所有实验通过Weights & Biases (wandb)进行跟踪，统一项目名称为：`ledpo_fullscale_experiment`。可以访问wandb网站查看实时训练曲线、指标和其他可视化内容。