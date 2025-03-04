# DPO基线训练实现

这个项目提供了一个完整的Direct Preference Optimization (DPO) 训练基线实现，用于从人类偏好数据中优化语言模型。项目将DPO训练过程拆分为前向传播、损失计算和反向传播三个阶段，便于深入理解DPO工作原理，为开发自定义的偏好优化算法（如FooDPO）做准备。

## 项目结构

```
dpo_baseline/
├── 开发日志.md           # 记录开发过程和问题解决
├── dpo_theory.md        # DPO算法理论基础和分析
├── dpo_trainer.py       # DPO训练器核心实现
├── train_dpo.py         # DPO训练脚本
├── evaluate_dpo.py      # DPO评估脚本
├── run_dpo.sh           # 一键运行DPO训练的脚本
└── README.md            # 项目说明文档
```

## 特性

- **完整的DPO实现**：包含完整的DPO算法实现，从数据处理到模型训练
- **训练过程拆解**：将训练过程拆分为前向传播、损失计算和反向传播阶段
- **详细的中间结果输出**：提供丰富的日志和可视化，便于理解DPO工作原理
- **本地小模型支持**：适配本地小型模型（如Qwen1.5-0.5B）进行快速实验
- **全中文注释**：代码中包含详细的中文注释，便于理解

## 快速开始

### 环境准备

需要安装以下依赖：

```bash
pip install torch transformers datasets matplotlib tqdm
```

### 运行训练

使用以下命令运行DPO训练：

```bash
python train_dpo.py \
    --model_path /root/models/Qwen1.5-0.5B \
    --data_path "Anthropic/hh-rlhf" \
    --output_dir ./dpo_output \
    --batch_size 4 \
    --epochs 3 \
    --beta 0.1 \
    --debug
```

### 运行评估

训练完成后，可以使用以下命令评估模型：

```bash
python evaluate_dpo.py \
    --base_model_path /root/models/Qwen1.5-0.5B \
    --dpo_model_path ./dpo_output/final-model \
    --eval_data_path "Anthropic/hh-rlhf" \
    --output_dir ./dpo_eval \
    --max_samples 100
```

### 使用一键运行脚本

也可以使用提供的一键运行脚本：

```bash
bash run_dpo.sh
```

## 如何扩展

1. **自定义损失函数**：修改`dpo_trainer.py`中的`compute_loss`方法
2. **调整奖励计算**：修改奖励相关的计算逻辑
3. **添加新的评估指标**：在`evaluate_dpo.py`中添加新的评估逻辑
4. **实现FooDPO**：基于当前实现，添加特定于食品领域的偏好表示和训练逻辑

## 参考资料

- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Hugging Face RLHF教程](https://huggingface.co/blog/rlhf)
- [LLaMA-Factory项目](https://github.com/hiyouga/LLaMA-Factory) 