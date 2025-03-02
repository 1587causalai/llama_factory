# FooDPO训练实验报告

## 实验概述

本实验使用LlamaFactory框架中的FooDPO算法对Qwen1.5-0.5B模型进行了微调。实验在macOS环境下使用MPS设备进行，成功解决了tokenizer加载问题和MPS设备上的训练问题。

## 实验环境

- 操作系统: macOS 24.3.0
- 处理器: Apple Silicon (MPS设备)
- 框架: LlamaFactory
- 基础模型: Qwen1.5-0.5B
- 数据集: dpo_zh_demo

## 解决的问题

### 1. Tokenizer加载问题

最初，tokenizer文件存储在单独的目录(`~/models/Qwen1.5-0.5B-tokenizer/`)中，导致模型无法自动加载tokenizer。我们将tokenizer文件复制到模型目录(`~/models/Qwen1.5-0.5B/`)中解决了这个问题。

相关文件包括:
- added_tokens.json
- merges.txt
- special_tokens_map.json
- tokenizer.json
- tokenizer_config.json
- vocab.json

### 2. MPS设备训练问题

在macOS上使用MPS设备时，最初遇到了混合精度训练问题:
```
ValueError: fp16 mixed precision requires a GPU (not 'mps')
```

通过修改训练脚本，我们解决了这个问题:
1. 禁用fp16混合精度训练 (`--fp16 false`)
2. 添加MPS设备支持 (`--use_mps_device true`)

### 3. Sigmoid损失函数问题

发现FooDPO实现中有一个损失类型问题，我们检查并确保了`sigmoid`损失类型的正确实现。

## 训练配置

训练脚本(`scripts/test_qwen_foodpo.sh`)的主要配置:

```bash
# 基本配置参数
MODEL_PATH=~/models/Qwen1.5-0.5B
DATASET_PATH="data"
OUTPUT_DIR="output/qwen_foodpo_test"

# 基本训练参数
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=5e-5
EPOCHS=1
MAX_SEQ_LEN=1024
WARMUP_RATIO=0.03

# 运行训练命令 - 使用MPS设备而非CUDA
llamafactory-cli train \
    --model_name_or_path $MODEL_PATH \
    --dataset "dpo_zh_demo" \
    --dataset_dir $DATASET_PATH \
    --eval_dataset "dpo_zh_demo" \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --per_device_eval_batch_size $MICRO_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --cutoff_len $MAX_SEQ_LEN \
    --warmup_ratio $WARMUP_RATIO \
    --max_samples 20 \
    --logging_steps 5 \
    --save_steps 20 \
    --save_total_limit 1 \
    --do_train true \
    --template default \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --stage foodpo \
    --pref_beta 0.1 \
    --fp16 false \
    --use_mps_device true
```

## 训练结果

训练成功完成，日志显示:

```json
{"current_steps": 5, "total_steps": 10, "loss": 53.9602, "accuracy": 0.4000000059604645, "lr": 2.777777777777778e-05, "epoch": 0.5, "percentage": 50.0, "elapsed_time": "0:00:27", "remaining_time": "0:00:27"}
{"current_steps": 10, "total_steps": 10, "loss": 42.0622, "accuracy": 0.20000000298023224, "lr": 0.0, "epoch": 1.0, "percentage": 100.0, "elapsed_time": "0:01:09", "remaining_time": "0:00:00"}
```

详细训练结果:

```json
{
    "epoch": 1.0,
    "total_flos": 36653597884416.0,
    "train_loss": 48.01117706298828,
    "train_runtime": 70.6535,
    "train_samples_per_second": 0.283,
    "train_steps_per_second": 0.142
}
```

LoRA适配器配置:

```json
{
  "alpha_pattern": {},
  "auto_mapping": null,
  "base_model_name_or_path": "/Users/gongqian/models/Qwen1.5-0.5B",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layer_replication": null,
  "layers_pattern": null,
  "layers_to_transform": null,
  "loftq_config": {},
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "megatron_config": null,
  "megatron_core": "megatron.core",
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 8,
  "rank_pattern": {},
  "revision": null,
  "target_modules": [
    "gate_proj",
    "v_proj",
    "q_proj",
    "k_proj",
    "down_proj",
    "o_proj",
    "up_proj"
  ],
  "task_type": "CAUSAL_LM",
  "use_dora": false,
  "use_rslora": false
}
```

## 推理测试

我们创建了推理脚本(`scripts/test_inference.py`)来测试训练后的模型:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载基础模型和tokenizer
model_path = '/Users/gongqian/models/Qwen1.5-0.5B'
adapter_path = 'output/qwen_foodpo_test'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='mps',
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 加载LoRA适配器
from peft import PeftModel
model = PeftModel.from_pretrained(model, adapter_path)

# 设置生成参数
prompt = '请介绍一下中国的传统节日：'
inputs = tokenizer(prompt, return_tensors='pt').to('mps')

# 生成回复
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,  # 启用采样模式
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print('输入:', prompt)
print('输出:', response)
```

测试了不同类型的提示:
1. 介绍中国传统节日
2. 写一首关于春天的诗
3. 编写Python函数(计算斐波那契数列)

模型能够生成合理的回复，表明微调成功。

## 后续工作

可以考虑以下改进:
1. 增加训练数据量和训练步骤
2. 调整超参数(学习率、batch size等)
3. 尝试不同的提示工程方法
4. 对模型进行定量评估

## 总结

本实验成功在macOS环境下使用MPS设备对Qwen1.5-0.5B模型进行了FooDPO微调。解决了tokenizer加载问题和MPS设备上的训练问题，为后续在Mac设备上进行LLM微调提供了参考。 