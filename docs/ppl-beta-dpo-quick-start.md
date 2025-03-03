# FooDPO: 基于困惑度的动态 Beta DPO 算法实现文档

## 1. 概述

FooDPO 是对标准 DPO（Direct Preference Optimization）算法的一种变体实现，它引入了一个关键创新：**基于输入困惑度（Perplexity）动态调整 beta 参数**。这种方法通过为不同复杂度的输入自适应地调整对齐强度，有望提高偏好对齐的效果和稳定性。

## 2. 算法原理

### 2.1 标准 DPO 公式

标准 DPO 损失函数定义为：

$$\mathcal{L}_{\text{DPO}}(\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \left[ \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right] \right) \right]$$

其中 $\beta$ 是一个固定的超参数，控制对齐强度。

### 2.2 FooDPO 公式创新

FooDPO 将固定的 $\beta$ 替换为依赖于输入 $x$ 的函数 $\beta(x)$：

$$\beta(x) = c \cdot \log(\text{PPL}(x)) \cdot \beta$$

其中：
- $\text{PPL}(x)$ 是模型对输入提示 $x$ 的困惑度
- $c$ 是一个缩放系数，通过 `pref_beta_scale` 参数控制
- $\beta$ 是基础 beta 参数

这种设计的直觉是：**对于模型认为更复杂（高困惑度）的输入，应该施加更强的对齐压力**。

## 3. 实现细节

FooDPO 的实现主要包括三个关键部分：

### 3.1 困惑度计算

在 `calculate_prompt_perplexity` 方法中，通过以下步骤计算输入提示的困惑度：

1. 提取提示部分（非回答部分）的 token
2. 计算这些 token 的平均负对数概率
3. 取指数得到困惑度

```python
# 计算平均log概率和困惑度
avg_log_probs = token_log_probs.sum(dim=-1) / seq_lengths
perplexity = torch.exp(-avg_log_probs)
```

### 3.2 动态 Beta 计算

在 `compute_preference_loss` 方法中，根据困惑度计算动态 beta：

```python
# 计算动态beta: β(x) = c · log(PPL(x)) · β
if perplexity is not None and hasattr(self.finetuning_args, "pref_beta_scale") and self.finetuning_args.pref_beta_scale > 0:
    # 增加数值稳定性
    log_ppl = torch.log(torch.clamp(perplexity, min=1.0))
    dynamic_beta = self.finetuning_args.pref_beta_scale * log_ppl * base_beta
else:
    dynamic_beta = base_beta * torch.ones_like(policy_chosen_logps)
```

### 3.3 指标监控

在 `get_batch_loss_metrics` 方法中，添加了对困惑度、对数困惑度和动态 beta 的监控：

```python
# 添加困惑度指标
metrics[f"{prefix}prompt_perplexity"] = prompt_perplexity.mean().item()

# 添加log困惑度指标
log_ppl = torch.log(torch.clamp(prompt_perplexity, min=1.0)).mean().item()
metrics[f"{prefix}log_prompt_perplexity"] = log_ppl

# 计算并记录动态beta
if hasattr(self.finetuning_args, "pref_beta_scale") and self.finetuning_args.pref_beta_scale > 0:
    dynamic_beta_value = self.finetuning_args.pref_beta_scale * log_ppl * self.beta
    metrics[f"{prefix}dynamic_beta"] = dynamic_beta_value
else:
    metrics[f"{prefix}dynamic_beta"] = self.beta
```

## 4. 使用指南

### 4.1 参数配置

要使用 FooDPO，需要在训练命令中指定以下关键参数：

- `--stage foodpo`: 使用 FooDPO 算法
- `--pref_beta 0.1`: 设置基础 beta 值
- `--pref_beta_scale 0.5`: 设置困惑度缩放系数

### 4.2 示例脚本

```bash
llamafactory-cli train \
    --model_name_or_path MODEL_PATH \
    --dataset "dpo_dataset" \
    --stage foodpo \
    --pref_beta 0.1 \
    --pref_beta_scale 0.5 \
    ... # 其他参数
```

### 4.3 训练监控

训练过程中可以关注以下关键指标：

- `prompt_perplexity`: 原始困惑度
- `log_prompt_perplexity`: 对数困惑度
- `dynamic_beta`: 实际使用的动态 beta 值
- `loss`: 训练损失值

这些指标可以通过日志或 WandB 仪表板观察。

## 5. 参数调优建议

### 5.1 pref_beta_scale 参数

`pref_beta_scale` 参数决定了困惑度对 beta 值的影响程度：

- **较低的值** (0.1-0.3): 困惑度对 beta 的影响较小，算法行为接近标准 DPO
- **中等的值** (0.4-0.7): 推荐的起始值，提供适度的动态调整
- **较高的值** (0.8-1.0): 困惑度对 beta 的影响较大，可能导致训练不稳定

### 5.2 pref_beta 参数

基础 `pref_beta` 参数设置与标准 DPO 类似：

- **较低的值** (0.05-0.1): 对参考模型行为的约束较小，模型更自由
- **较高的值** (0.2-0.5): 对参考模型行为的约束较大，可能导致过拟合

## 6. 实验观察与分析

通过启用 log_prompt_perplexity 和 dynamic_beta 的监控，可以进行以下分析：

1. **困惑度与输入复杂度的关系**：观察不同类型输入的困惑度分布
2. **动态 beta 与训练稳定性**：分析 dynamic_beta 与损失的相关性
3. **模型性能比较**：对比固定 beta 和动态 beta 在不同测试集上的表现

## 7. 未来改进方向

1. **自适应缩放策略**：探索更复杂的 beta 调整策略，如根据训练进度动态调整 pref_beta_scale
2. **多维度困惑度**：考虑分别计算和使用提示和回答的困惑度
3. **与其他技术结合**：探索与 KTO、ORPO 等其他偏好对齐方法的结合

## 8. 结论

FooDPO 算法通过引入基于困惑度的动态 beta 机制，为 DPO 训练提供了一种更灵活的方法。这种方法有潜力在不同复杂度的输入上实现更平衡的偏好对齐，提高模型的整体表现。通过持续监控和分析 log_prompt_perplexity 和 dynamic_beta 指标，可以更深入地理解和优化这种动态对齐机制。
