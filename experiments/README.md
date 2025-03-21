# LLamaFactory 实验目录

本目录包含多种实验配置和脚本，用于对比不同的训练方法和模型设置。

## 可用实验

### DPO vs LEDPO 完整对比实验 (`dpo_vs_ledpo_full`)

全面对比标准DPO和可学习Beta的DPO (LEDPO) 在大规模偏好数据上的效果差异。使用了更完整的数据集、更长的训练时间和更严格的评估方法。

运行方法:
```bash
bash experiments/dpo_vs_ledpo_full/run_experiment.sh
```

详情请参考 [dpo_vs_ledpo_full/README.md](dpo_vs_ledpo_full/README.md)

### DPO vs LEDPO 演示实验 (`dpo_vs_ledpo_demo`)

一个简化版的DPO与LEDPO对比实验，使用较小的数据集和更短的训练时间，适合快速测试和演示。

运行方法:
```bash
bash experiments/dpo_vs_ledpo_demo/run_experiment.sh
```

### 冻结策略实验 (`frozen_policy`)

对比冻结策略模型与非冻结策略模型在偏好优化中的差异。

### 冻结策略演示实验 (`frozen_policy_demo`)

冻结策略的简化演示版本。

## 如何添加新实验

要添加新的实验，请创建一个新的子目录，并包含以下文件:

1. `README.md` - 描述实验目标、配置和预期结果
2. 配置文件 (通常为YAML格式)
3. `run_experiment.sh` - 运行实验的脚本
4. `analyze_logs.py` - 分析实验结果的脚本

## 建议的实验流程

1. 确定实验目标和假设
2. 准备模型和数据集
3. 创建实验配置文件
4. 运行实验
5. 分析结果
6. 记录发现和结论






