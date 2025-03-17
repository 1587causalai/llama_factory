# 冻结与非冻结策略模型对比实验

本实验旨在验证策略模型冻结功能和动态beta的学习行为，通过对比冻结策略模型与非冻结策略模型的训练过程，分析两种情况下delta值、beta值的变化趋势。

## 实验设置

* **冻结策略模型实验**: 启用`freeze_policy: true`，冻结策略模型参数，仅beta_head可训练
* **非冻结策略模型实验**: 设置`freeze_policy: false`，策略模型和beta_head同时训练

两组实验使用相同的：
- 模型: Qwen1.5-0.5B
- 数据集: dpo_en_demo
- 学习率: 1.0e-4
- 训练轮次: 3

## 运行实验

```bash
# 方法1: 一次性运行两个实验并生成对比图表
cd /path/to/llama_factory
./experiments/frozen_policy_demo/run_experiment.sh

# 方法2: 单独运行实验
# 运行冻结模型实验
python ledpo_progressive_dev/run_train_and_plot.py --config experiments/frozen_policy_demo/frozen_config.yaml --wandb_project frozen_policy_demo

# 运行非冻结模型实验
python ledpo_progressive_dev/run_train_and_plot.py --config experiments/frozen_policy_demo/unfrozen_config.yaml --wandb_project frozen_policy_demo

# 生成对比图表
python experiments/frozen_policy_demo/compare_results.py
```

## 结果分析

实验完成后，可以查看以下结果：

1. 各组实验原始结果:
   - 冻结模型: `experiments/frozen_policy_demo/results/frozen/`
   - 非冻结模型: `experiments/frozen_policy_demo/results/unfrozen/`

2. 对比分析结果:
   - 全指标对比图: `experiments/frozen_policy_demo/comparison/frozen_vs_unfrozen_comparison.png`
   - Delta和Beta摘要图: `experiments/frozen_policy_demo/comparison/delta_beta_summary.png`

## 核心关注点

1. **冻结模型下的delta值趋势** - 理论上，当策略模型被冻结时，delta值应该保持相对稳定
2. **beta学习行为差异** - 比较两种情况下beta值的学习路径和收敛趋势
3. **正负样本beta差异** - 分析pos_beta和neg_beta在两种情况下的变化

## 预期结果

1. 冻结策略模型时，delta值应保持相对稳定，beta_head能够有效学习
2. 非冻结策略模型时，delta值和beta值可能同时变化，相互影响
3. 验证delta和beta值之间的理论关系 