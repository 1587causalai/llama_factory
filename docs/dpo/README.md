# DPO训练调试文档与代码

本目录包含了DPO（Direct Preference Optimization）训练相关的调试文档和代码。这些资源专注于帮助理解DPO训练流程、调试技巧以及与Weights & Biases (wandb)的集成机制。

## 主要内容

```bash
python dpo_baseline/run_ledpo_debug.py examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml
```


### 1. 调试脚本
- [`dpo_baseline/run_ledpo_debug.py`](../../dpo_baseline/run_ledpo_debug.py) - 专为断点调试设计的DPO训练脚本，




### 2. 技术文档
- [`debug_guide.md`](debug_guide.md) - DPO训练调试指南，详细说明调试脚本的使用方法和关键功能
- [`wandb_callbacks.md`](wandb_callbacks.md) - 详细解释Weights & Biases报告机制的内部工作原理

### 3. 配置文件
- [`examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml`](../../examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml) - 使用Qwen1.5-0.5B模型的LEDPO训练配置

## 使用指南

### 调试脚本使用方法
```bash
# 激活适当的conda环境
conda activate llama # 或其他适当的环境

# 运行调试脚本
python dpo_baseline/run_ledpo_debug.py examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml
```

### 关键调试断点
在`run_ledpo_debug.py`脚本中，关键断点位置已用注释标记：
```python
# BREAKPOINT: [说明]
# pdb.set_trace()
```

要启用断点调试，只需取消相应行的注释，然后运行脚本。详细的调试指南请参见[debug_guide.md](debug_guide.md)。

### Wandb集成
训练配置中已包含wandb集成，您可以通过以下方式自定义：
1. 在配置文件中设置`report_to: [wandb]`
2. 通过环境变量设置wandb项目名称：`export WANDB_PROJECT="your_project_name"`
3. 首次使用前需登录wandb：`wandb login`

关于wandb报告机制的详细解释，请参见[wandb_callbacks.md](wandb_callbacks.md)。 