# LEDPO - 可学习Beta的DPO训练算法

LEDPO (Learnable Beta Direct Preference Optimization) 是对标准DPO的改进版本，主要特点是引入了可学习的beta参数。

## 简介

标准DPO算法使用固定的beta值作为损失函数中的超参数，而LEDPO允许模型自适应地学习和调整beta参数，从而优化训练过程。

## 使用方法

在配置文件中指定stage为"ledpo"：

```yaml
stage: ledpo
pref_beta: 0.1  # 初始beta值
pref_loss: sigmoid  # 可选: sigmoid (dpo), orpo, simpo
```

## 训练命令

```bash
# 使用llamafactory-cli启动训练
llamafactory-cli train examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml

# 或使用调试脚本启动
python dpo_baseline/run_ledpo_debug.py examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml
```

## 核心文件

- `src/llamafactory/train/ledpo/trainer.py` - 训练器实现
- `src/llamafactory/train/ledpo/workflow.py` - 训练流程定义
- `dpo_baseline/run_ledpo_debug.py` - 调试用训练脚本
- `examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml` - 示例配置 