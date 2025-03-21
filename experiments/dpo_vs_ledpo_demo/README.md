# DPO vs LEDPO 对比实验

本实验旨在对比标准DPO和可学习Beta的DPO (LEDPO) 在模型微调中的效果差异。

## 实验背景

DPO (Direct Preference Optimization) 是一种基于人类偏好数据直接优化模型的方法。标准DPO使用固定的beta值（通常为0.1）作为温度参数，而LEDPO (Learnable Beta DPO) 则允许模型自动学习每个训练样本的最优beta值，理论上能提高模型对人类偏好的学习效果。

## 实验目标

1. 对比标准DPO和LEDPO在训练效率上的差异
2. 分析LEDPO中beta值的学习过程和最终分布
3. 验证动态beta是否能提升模型性能

## 实验配置

- **基础模型**: Qwen1.5-0.5B
- **微调方法**: LoRA (rank=8)
- **训练数据集**: dpo_en_demo
- **评估数据集**: dpo_zh_demo
- **训练参数**:
  - 批次大小: 2
  - 梯度累积步数: 4
  - 学习率: 1.0e-4
  - 训练轮次: 1.0
  - 优化器: AdamW
  - 学习率调度: cosine

## 实验方案

实验分为两组:

1. **标准DPO** - 使用固定的beta值(0.1)
2. **LEDPO** - 使用可学习的动态beta值

## 运行实验

完整运行:

```bash
bash experiments/dpo_vs_ledpo_demo/run_experiment.sh
```

跳过部分实验:

```bash
# 只运行标准DPO
bash experiments/dpo_vs_ledpo_demo/run_experiment.sh --skip-ledpo

# 只运行LEDPO
bash experiments/dpo_vs_ledpo_demo/run_experiment.sh --skip-dpo
```

## 结果分析

运行分析脚本:

```bash
python experiments/dpo_vs_ledpo_demo/analyze_logs.py
```

分析脚本将生成以下内容:

1. 训练损失对比图
2. 评估准确率对比图
3. 奖励差值对比图
4. Beta值变化趋势图(仅LEDPO)
5. 奖励值对比图(Chosen vs Rejected)
6. 实验结果摘要报告

## 结果位置

- **标准DPO结果**: `experiments/dpo_vs_ledpo_demo/results/dpo`
- **LEDPO结果**: `experiments/dpo_vs_ledpo_demo/results/ledpo`
- **分析结果**: `experiments/dpo_vs_ledpo_demo/analysis`

## 预期结果

如果LEDPO工作正常，我们预期看到:

1. LEDPO比标准DPO有更低的训练损失和更高的评估准确率
2. LEDPO中beta值会随训练进行合理变化，而非趋于极端值
3. LEDPO对chosen和rejected样本的区分度更高(更大的reward margin) 