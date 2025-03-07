# DPO训练调试指南

## 调试脚本简介

`dpo_baseline/run_ledpo_debug.py`是一个专为调试DPO训练流程设计的脚本。它基于原有的`run_ledpo_rich.py`简化而来，移除了复杂的UI美化代码，专注于提供清晰的模块化结构和方便的断点调试位置。

## 脚本设计理念

1. **模块化结构**：将DPO训练流程分为明确的阶段，每个阶段对应一个函数
2. **断点友好**：在关键位置添加断点注释，方便调试
3. **简化输出**：保留必要的日志输出，但移除复杂的UI元素
4. **可追踪性**：提供清晰的日志和状态信息

## DPO训练七个关键阶段

脚本将DPO训练流程分为以下七个关键阶段：

1. **配置加载和处理**：`load_and_process_config()`
2. **模型和分词器准备**：`prepare_model_components()`
3. **数据准备**：`prepare_training_data()`
4. **训练器设置**：`setup_dpo_trainer()`
5. **执行训练**：`run_training()`
6. **执行评估**：在`run_training()`函数内部
7. **异常处理和日志记录**：在`run_dpo_workflow()`函数中

## 调试关键断点

脚本中已标记了以下关键断点位置：

```python
# 1. 配置文件加载相关断点
# BREAKPOINT: 配置加载前 - 检查配置文件路径
# BREAKPOINT: 配置加载后 - 检查解析后的参数对象

# 2. 模型加载相关断点
# BREAKPOINT: 加载tokenizer前
# BREAKPOINT: 检查tokenizer - 查看词表大小和特殊token
# BREAKPOINT: 检查模板 - 检查模板格式和结构
# BREAKPOINT: 检查模型 - 查看模型结构和参数
# BREAKPOINT: 检查参考模型 - 确认参考模型加载正确

# 3. 数据集相关断点
# BREAKPOINT: 数据集加载前 - 检查数据参数
# BREAKPOINT: 数据集加载后 - 检查数据集结构和样本
# BREAKPOINT: 检查数据整理器 - 确认数据整理逻辑正确

# 4. 训练器相关断点
# BREAKPOINT: 创建回调函数前
# BREAKPOINT: 初始化训练器前 - 检查回调函数
# BREAKPOINT: 初始化训练器后 - 检查训练器配置和优化器

# 5. 训练相关断点
# BREAKPOINT: 训练前 - 检查训练参数和模型状态
# BREAKPOINT: 训练后 - 检查训练结果和指标

# 6. 评估相关断点
# BREAKPOINT: 评估前 - 检查评估设置
# BREAKPOINT: 评估后 - 检查评估结果

# 7. 工作流相关断点
# BREAKPOINT: 工作流开始 - 整体流程开始前检查
# BREAKPOINT: 工作流结束 - 整体流程完成后检查
# BREAKPOINT: 错误处理 - 异常发生时检查
```

## 使用方法

1. **启用断点**：取消注释相应断点位置的`pdb.set_trace()`行
2. **运行脚本**：
   ```bash
   python dpo_baseline/run_ledpo_debug.py examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml
   ```
3. **在断点处调试**：
   - 使用`p 变量名`查看变量值
   - 使用`pp 变量名`美化打印变量
   - 使用`dir(对象)`查看对象属性和方法
   - 使用`n`执行下一行
   - 使用`s`步入函数
   - 使用`c`继续执行
   - 使用`q`退出调试

## 调试Wandb集成

本脚本特别适合调试Wandb的集成和报告机制。关键位置包括：

1. **ReporterCallback创建**：
   ```python
   callbacks.append(ReporterCallback(
       model_args=model_args,
       data_args=data_args,
       finetuning_args=finetuning_args,
       generating_args=generating_args
   ))
   ```

2. **指标记录**：
   ```python
   metrics = train_result.metrics
   trainer.log_metrics("train", metrics)
   trainer.save_metrics("train", metrics)
   ```

3. **评估指标记录**：
   ```python
   metrics = trainer.evaluate(metric_key_prefix="eval")
   trainer.log_metrics("eval", metrics)
   trainer.save_metrics("eval", metrics)
   ```

详细的Wandb报告机制解释请参见[wandb_callbacks.md](wandb_callbacks.md)文档。 