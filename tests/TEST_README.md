# LLaMA-Factory 测试脚本说明文档

本文档对 LLaMA-Factory 项目中的测试脚本进行分类说明，帮助开发者快速了解和使用这些测试脚本。

## 测试脚本分类

### 1. Beta-DPO 相关测试

#### 核心功能测试
- `test_beta_head.py`: Beta 头部模块的单元测试
  - 输出：Beta head 的前向传播结果和梯度
  - 预期：Beta 值应在 (0, 1) 范围内，梯度应该能正常回传
- `test_betadpo.py`: Beta-DPO 基础功能测试
  - 输出：模型损失值、Beta 值分布、PPO 比率
  - 预期：损失值应收敛，Beta 值分布合理
- `test_betadpo_forward_pass.py`: Beta-DPO 前向传播和损失计算的完整流程测试
  - 输出：chosen/rejected logits、Beta 值、PPL 值等详细指标
  - 预期：各项指标数值在合理范围内，无 NaN 或异常值

#### 逐步测试
- `test_stepbystep_betadpo.py`: Beta-DPO 训练流程的逐步测试
  - 输出：每个步骤的中间结果和状态
  - 预期：每个步骤都能正确执行，状态正确传递
- `test_stepbystep_betadpo_qwen_0.5b.py`: 使用 Qwen-0.5B 模型的 Beta-DPO 逐步测试
  - 输出：模型加载状态、训练过程指标
  - 预期：模型能正确加载和训练

### 2. DPO 相关测试

#### 核心功能测试
- `test_dpo.py`: DPO 基础功能测试
  - 输出：DPO 损失值、策略比率
  - 预期：损失收敛，策略比率在合理范围
- `test_dpo_qwen_0.5b.py`: 使用 Qwen-0.5B 模型的 DPO 测试
  - 输出：模型输出的 logits 和 loss
  - 预期：输出格式正确，数值稳定

#### 数据处理测试
- `test_dpo_data_processing.py`: DPO 数据处理流程测试
  - 输出：处理后的数据样本示例
  - 预期：数据格式符合要求，标签正确
- `test_dpo_data_processing_qwen.py`: Qwen 模型的 DPO 数据处理测试
  - 输出：tokenization 结果、attention mask
  - 预期：特殊 token 处理正确，长度对齐

#### 逐步测试
- `test_stepbystep_dpo_qwen_0.5b.py`: 使用 Qwen-0.5B 模型的 DPO 逐步测试
  - 输出：每个训练步骤的详细信息
  - 预期：训练过程平稳，指标变化合理

## 环境要求

### 基础环境
- Python 3.8+
- PyTorch 2.0+
- transformers 4.34+
- CUDA 11.7+ (GPU 训练)

### 预训练模型
可用模型路径：`/root/models/`
- interlm1.8b
- DeepSeek-R1-Distill-Qwen-1.5B

### Conda 环境
所有测试必须在 `llama` conda 环境下运行：
```bash
conda activate llama
```

## 目录结构说明

### 辅助目录
- `configs/`: 测试配置文件目录
- `data/`: 测试数据集目录
- `model/`: 测试模型存储目录
- `eval/`: 评估相关测试目录
- `e2e/`: 端到端测试目录
- `runs/`: 测试运行日志和输出目录

## 使用指南

### Beta-DPO 测试流程
1. 首先运行 `test_beta_head.py` 确保 Beta 头部模块正常工作
2. 使用 `test_betadpo.py` 验证 Beta-DPO 的基本功能
3. 运行 `test_betadpo_forward_pass.py` 检查完整的前向传播流程
4. 如需详细了解训练过程，可以运行 `test_stepbystep_betadpo.py` 或特定模型的逐步测试

### DPO 测试流程
1. 使用 `test_dpo.py` 验证 DPO 的基本功能
2. 运行 `test_dpo_data_processing.py` 检查数据处理流程
3. 如使用 Qwen 模型，可以运行对应的专用测试脚本
4. 需要详细了解训练过程时，可以运行逐步测试脚本

## 注意事项

1. 运行测试前请确保已激活 conda 环境：`conda activate llama`
2. 部分测试需要下载预训练模型，请确保网络连接正常
3. GPU 测试需要确保有足够的显存
4. 建议按照推荐的测试流程顺序执行测试
5. 每个测试都会输出至少一个样本示例，便于人工检查

## 常见问题

1. 如果遇到显存不足，可以尝试减小配置文件中的 batch_size
2. 数据处理相关的错误通常与数据集格式或预处理步骤有关
3. 模型加载错误可能是由于模型文件不完整或版本不匹配导致
4. 如果测试输出与预期不符，请检查：
   - 环境是否正确（conda 环境）
   - 数据格式是否正确
   - 模型路径是否正确
   - 配置参数是否合适

## 文档维护

本文档会随着测试脚本的添加和修改而更新。每次添加新的测试脚本或修改现有测试时，都需要：
1. 在相应分类下添加新的测试说明
2. 更新测试输出和预期结果说明
3. 如有必要，更新环境要求和注意事项
4. 添加新出现的常见问题和解决方案 