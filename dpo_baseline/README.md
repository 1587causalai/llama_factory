# DPO基线实现与调试脚本

本目录包含了DPO(Direct Preference Optimization)及其变种的实现脚本。

## 文件说明

- `run_ledpo_debug.py`: LEDPO (Learnable Beta DPO) 训练调试脚本，用于断点调试和理解训练流程

## 使用方法

### LEDPO训练

```bash
# 使用CLI接口启动训练
llamafactory-cli train examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml

# 或使用调试脚本进行训练
python dpo_baseline/run_ledpo_debug.py examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml
``` 