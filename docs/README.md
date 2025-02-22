# README

基于可学习 beta 值的 DPO (Direct Preference Optimization) 实现，用于大语言模型的人类偏好对齐。

## 项目概述

本项目旨在开发一个基于 Learnable Beta DPO 的人类偏好对齐微调框架，通过自适应调整 DPO 算法中的 β 参数来实现更精细的探索-利用平衡控制。

- **基础模型**: Qwen-0.5B/Qwen-1.5B/...
- **实际实现**: 使用 Qwen1.5-0.5B-Chat, deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B（基于 Qwen 的高效蒸馏版本）
- **创新点**: 设计了与策略模型紧密耦合的 `BetaHead` 网络，实现动态 β 值计算


## Learnable Beta DPO 数学理论

在标准的 DPO (Direct Preference Optimization)  算法中，一个关键的超参数是 $\beta$，它控制着模型在参考策略和奖励信息之间的权衡。从信息融合的角度来看：

- 较大的 $\beta$ 值使模型更倾向于遵循参考策略 $\pi_{\text{ref}}$，保持原有行为
- 较小的 $\beta$ 值使模型更多地利用奖励信息，进行策略调整

然而，传统的 DPO 通常使用固定的 $\beta$ 值，这带来了两个主要局限：

1. **上下文不敏感**：不同场景下可能需要不同的探索-利用权衡
   - 在模型熟悉的领域，应该更多地保持参考策略的行为
   - 在模型不熟悉的领域，应该更多地从奖励信息中学习

2. **优化效率受限**：固定的权衡策略可能导致
   - 在某些场景下过度保守，错过学习机会
   - 在某些场景下过度激进，损失已有能力

这种"一刀切"的方式无法针对不同的上下文动态调整学习策略，限制了模型在复杂、多变场景下的优化效果。

这是一个基于可学习 beta 值的 DPO 实现。项目的核心思想是设计一个可学习的函数：

$$\beta(x) = w \cdot \log(PPL(x)) \cdot f(x)$$

其中：
- $w$ 是一个可学习的参数
- $\mathrm{PPL}(x)$ 是上下文 $x$ 的困惑度, PPL 越大，表示模型越困惑，越难预测下一个词。使用策略模型 $\pi_\theta$ 计算, $\mathrm{PPL}_{\pi_\theta}(x)$ 反映了**策略模型对输入的确定性程度**。对于给定的输入序列 $x = (x_1, x_2, ..., x_m)$，困惑度定义为：
$$\mathrm{PPL}_{\pi_\theta}(x) = \exp \left( - \frac{1}{m} \sum_{i=1}^m \log \pi_\theta(x_i | x_{<i}) \right)$$


- $f(x)$ 是上下文 $x$ 的函数，其取值范围为 $[1-\epsilon, 1+\epsilon]$, 具体实现中，
$$f(x) = 1 + \epsilon \cdot \tanh(NN(h_{\pi_\theta}(x)))$$  
其中 $h_{\pi_\theta}(x)$ 是由策略模型 $\pi_\theta$ 得到的最后一层隐状态，$NN(h_{\pi_\theta}(x))$ 是一个神经网络. 



我的需求是要为 LLM (e.g.  Qwen-0.5B) 新增一个 betahead 层，我们想基于 llama-factory 库来实现这个功能。
