# 冻结策略模型与非冻结模型对比实验

本实验旨在验证LEDPO (可学习Beta DPO) 算法中冻结策略模型的效果，特别是分析beta_head参数更新对delta值的影响。

## 实验目的

1. 验证beta_head是否能够在策略模型冻结的情况下进行有效学习
2. 对比冻结策略模型与正常训练模型的delta和beta值分布差异
3. 分析beta值与delta值之间的相关性

## 实验设置

- **冻结模型**: 冻结策略模型参数，仅训练beta_head部分
- **冻结模型-diff-ref模型**: 冻结策略模型参数，policy 和 ref model 不同, 训练beta_head部分
- **非冻结模型**: 同时训练策略模型和beta_head部分
- **模型基础**: Qwen1.5-0.5B
- **数据集**: hh_rlhf_en
- **训练轮次**: 3轮
- **样本数量**: 2000

## 文件说明

- `frozen_config.yaml`: 冻结策略模型配置
- `frozen_config_diff_ref.yaml`: 冻结策略模型-不同参考模型配置
- `unfrozen_config.yaml`: 非冻结策略模型配置
- `run_experiment.sh`: 运行实验的脚本
- `analyze_logs.py`: 分析实验结果的脚本
- `results/`: 训练结果存放目录
- `analysis/`: 分析结果存放目录

## 使用方法

1. 运行完整实验:

```bash
cd path/to/llama_factory
bash experiments/frozen_policy/run_experiment.sh
```

2. 单独运行冻结模型实验:

```bash
python ledpo_progressive_dev/run_train_and_plot.py --config experiments/frozen_policy/frozen_config.yaml --wandb_project ledpo_frozen_policy

# different ref model
python ledpo_progressive_dev/run_train_and_plot.py --config experiments/frozen_policy/frozen_config_diff_ref.yaml --wandb_project ledpo_frozen_policy
```

3. 单独运行非冻结模型实验:

```bash
python ledpo_progressive_dev/run_train_and_plot.py --config experiments/frozen_policy/unfrozen_config.yaml --wandb_project ledpo_frozen_policy
```

4. 单独运行分析脚本:

```bash
# 分析两个模型（冻结和非冻结）
python experiments/frozen_policy/analyze_logs.py \
  --frozen_dir experiments/frozen_policy/results/frozen \
  --unfrozen_dir experiments/frozen_policy/results/unfrozen \
  --output_dir experiments/frozen_policy/analysis

# 分析三个模型（冻结、冻结-新ref和非冻结）
python experiments/frozen_policy/analyze_logs.py \
  --frozen_dir experiments/frozen_policy/results/frozen \
  --frozen_diff_ref_dir experiments/frozen_policy/results/frozen_diff_ref \
  --unfrozen_dir experiments/frozen_policy/results/unfrozen \
  --output_dir experiments/frozen_policy/analysis
```

## 预期结果

1. 冻结模型:
   - delta值应保持接近于0，因为策略模型参数被冻结不变
   - beta值应该有变化，表明beta_head正在学习
   - beta和delta之间相关性较弱

2. 冻结模型-不同参考模型:
   - delta值应保持接近于0，但可能与标准冻结模型略有不同
   - beta值变化模式可能与标准冻结模型不同
   - 可探究不同参考模型对学习动态的影响

3. 非冻结模型:
   - delta值应该有明显变化，并且均值为正，表明模型在学习偏好
   - beta值变化应与delta值表现出一定的相关性
   - beta和delta之间应有一定的相关性

## 结果分析

实验完成后，分析脚本将生成以下分析结果:

1. beta值和delta值的对比图表
2. beta与delta的散点图和相关性分析
3. 统计指标汇总图表
4. 详细的markdown格式分析报告

这些结果将保存在`experiments/frozen_policy/analysis/`目录下。 