# LLaMA Factory与EasyR1项目关系说明

## 项目关系概述

[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 和 [EasyR1](https://github.com/hiyouga/EasyR1) 是由同一研发团队（主要是hiyouga）开发的两个互补项目，它们共同构成了一个全面的大模型训练和优化生态系统。

- **LLaMA Factory**：专注于提供一个统一的高效微调框架，支持100+大语言模型和视觉语言模型的训练
- **EasyR1**：特别针对强化学习训练领域，提供高效、可扩展的多模态强化学习训练框架

## 项目特点对比

### LLaMA Factory
- **主要算法**：SFT, DPO, KTO, PPO, RM, PT等
- **侧重点**：通用微调框架，适用于各种模型和任务
- **特点**：支持模型广泛，使用简单，文档完善
- **适用场景**：通用微调、特定任务调优、模型评估等

### EasyR1
- **主要算法**：GRPO（以及后续将支持的其他RL算法）
- **侧重点**：高效的强化学习训练，特别是多模态模型
- **特点**：基于veRL项目，优化了大规模强化学习训练效率
- **适用场景**：大规模模型的RLHF训练，多模态模型优化等

## GRPO算法在EasyR1中的实现

GRPO (Group Relative Policy Optimization) 是EasyR1项目中的核心算法之一，它是对传统PPO算法的创新改进，具有以下特点：

1. **高效内存使用**：通过群组相对优势估计取代传统值函数，大幅降低内存开销
2. **模型支持**：
   - Qwen2/Qwen2.5语言模型
   - Qwen2/Qwen2.5-VL视觉语言模型
   - DeepSeek-R1蒸馏模型
3. **实用性**：易于使用的训练脚本和示例数据集格式

## 为什么需要两个项目

LLaMA Factory和EasyR1之所以分为两个不同的项目，主要有以下几个原因：

1. **关注点不同**：
   - LLaMA Factory关注广泛模型支持和通用微调算法
   - EasyR1专注于高效强化学习训练和多模态支持
   
2. **底层架构不同**：
   - EasyR1基于veRL项目，针对RLHF做了特殊优化
   - LLaMA Factory采用更通用的架构，适合各类微调方法
   
3. **资源优化策略不同**：
   - EasyR1特别优化了大规模模型训练的效率和资源利用
   - LLaMA Factory侧重于通用性和易用性

## 如何选择适合的项目

### 选择LLaMA Factory的场景：
- 需要进行常规的微调任务（如SFT, DPO等）
- 需要广泛的模型和任务支持
- 简单的训练设置和详细的文档需求
- 资源受限环境下的高效训练

### 选择EasyR1的场景：
- 需要使用GRPO等高级强化学习算法
- 训练大规模模型（如7B以上）
- 多模态模型（特别是视觉-语言模型）训练
- 对训练效率和资源利用有极高要求

## 未来发展

随着两个项目的发展，未来可能会有更多的功能交叉和互补，例如：

1. EasyR1中的高效RLHF算法可能会被引入LLaMA Factory
2. LLaMA Factory中的广泛模型支持可能会被EasyR1采纳
3. 两个项目可能会共享更多的代码库和最佳实践

## 相关资源

- LLaMA Factory GitHub: [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- EasyR1 GitHub: [https://github.com/hiyouga/EasyR1](https://github.com/hiyouga/EasyR1)
- GRPO算法概述: [grpo_overview.md](grpo_overview.md) 