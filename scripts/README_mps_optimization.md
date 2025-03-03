# Apple Silicon MPS设备上LLM训练内存优化指南

## 问题描述

在Apple Silicon芯片上使用MPS（Metal Performance Shaders）后端进行大语言模型训练时，经常会遇到内存不足的问题，错误信息通常为：

```
RuntimeError: MPS backend out of memory (MPS allocated: xx.xx GB, other allocations: xxx.xx MB, max allowed: xx.xx GB). 
Tried to allocate x.xx GB on private pool.
```

## 优化措施

我们通过以下方法成功在Qwen1.5-0.5B模型上运行了fooDPO训练，有效避免了内存溢出问题：

### 1. 环境变量设置

```bash
# 设置MPS内存使用限制
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

这个设置会禁用MPS后端的内存限制，允许使用更多的可用内存，但要注意可能导致系统不稳定。

### 2. 减小序列长度

- 将`MAX_SEQ_LEN`从1024减少到512或更小
- 序列长度对内存使用影响很大，根据我们的测试，每减少一半的序列长度，内存消耗会显著降低

### 3. 减小批量大小和梯度累积

- 使用小的`per_device_train_batch_size`（如1）
- 适当减少`gradient_accumulation_steps`，降低内存压力

### 4. 启用梯度检查点

```bash
--gradient_checkpointing true
```

启用梯度检查点可以大幅减少前向传播时存储的激活值，以时间换空间，是一种很有效的内存优化方法。

### 5. 降低LoRA参数

- 将`lora_rank`从8减少到4
- 减少`save_total_limit`，只保留少量检查点

### 6. 限制数据量

```bash
--max_samples 10
```

特别针对小数据集训练，可以进一步限制样本数量，降低内存压力。

### 7. 其他优化技巧

- 设置`dataloader_num_workers=0`，避免多进程导致的MPS设备问题
- 禁用`group_by_length`，减少内存碎片
- 训练前清理输出目录，避免旧模型文件占用空间
- 关闭其他内存密集型应用程序
- 训练期间监控内存使用情况

## 优化脚本示例

我们在`scripts/test_qwen_foodpo.sh`中实施了上述优化措施：

```bash
# 设置MPS内存使用限制
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# 基本配置参数
MODEL_PATH=~/models/Qwen1.5-0.5B
DATASET_PATH="data"
OUTPUT_DIR="output/qwen_foodpo_test"

# 训练参数 - 适合小数据集且优化内存使用
MICRO_BATCH_SIZE=1            # 小批量大小，避免梯度更新过大
GRADIENT_ACCUMULATION_STEPS=1 # 减少梯度累积步数，降低内存压力
LEARNING_RATE=1e-4            # 提高学习率以便在少量数据上快速学习
EPOCHS=3                      # 增加轮次以便在小数据集上充分学习
MAX_SEQ_LEN=512               # 减小序列最大长度以减少内存使用

# ... 其他配置 ...

# 运行训练命令
llamafactory-cli train \
    # ... 其他参数 ...
    --cutoff_len $MAX_SEQ_LEN \
    --max_samples 10 \
    --save_total_limit 1 \
    --lora_rank 4 \
    --dataloader_num_workers 0 \
    --gradient_checkpointing true
```

## 诊断工具

我们还创建了一个诊断脚本`scripts/mps_memory_debug.py`，可以帮助您分析MPS设备上的内存使用情况，包括：

- 系统内存信息
- MPS设备配置
- 模型加载内存消耗
- 不同LoRA配置的内存影响
- 不同序列长度的内存影响

使用方法：
```bash
python scripts/mps_memory_debug.py --model_path ~/models/Qwen1.5-0.5B
```

## 结论

通过适当的内存优化，我们成功在Apple Silicon设备上运行了LLaMA Factory的fooDPO训练。这些优化不仅适用于fooDPO，也适用于其他训练方法，如标准的SFT和DPO训练。

对于更大的模型，可能需要更激进的优化措施，甚至考虑使用更小的模型变体或使用量化技术来降低内存需求。

最后，随着PyTorch MPS后端的不断改进，未来这些内存问题可能会得到更好的解决。 