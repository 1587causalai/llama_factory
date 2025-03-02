# DPO (Direct Preference Optimization) 前向传播分析

## 1. 算法原理概述

Direct Preference Optimization (DPO) 是一种偏好学习算法，通过利用人类偏好数据直接优化语言模型。不同于传统的强化学习方法（如PPO），DPO避免了显式的奖励模型和策略优化步骤，而是将人类偏好直接转化为对模型参数的优化目标。

DPO的核心思想是基于以下观察：给定参考模型和一组偏好数据（由选中回复和拒绝回复组成），我们可以推导出一个目标函数，使模型学习产生更符合人类偏好的输出。

## 2. 数据流和输入格式

DPO算法需要的数据格式与SFT不同，它要求输入的数据包含三个部分：

1. **提示 (prompt)**: 原始用户输入
2. **选择的回复 (chosen)**: 人类偏好的、更好的回复
3. **拒绝的回复 (rejected)**: 质量较差的回复

这些数据通常以这样的格式组织：
```json
{
  "prompt": "请解释机器学习中的梯度下降算法",
  "chosen": "梯度下降是一种优化算法...(高质量回复)",
  "rejected": "梯度下降就是一种算法...(低质量回复)"
}
```

### 2.1 数据处理

在LLaMA Factory中，这些数据通过特殊的数据整理器（`PairwiseDataCollatorWithPadding`）处理，将一个批次的数据转换为以下格式：

```
[prompt+chosen_1, prompt+rejected_1, prompt+chosen_2, prompt+rejected_2, ...]
```

这使得前向传播过程可以同时处理选中和拒绝的样本，计算它们各自的对数概率。

## 3. 前向传播实现

DPO的前向传播流程主要由`CustomDPOTrainer`类中的`concatenated_forward`和`get_batch_loss_metrics`方法实现。

### 3.1 总体流程

1. 输入数据被组织为交替的选中/拒绝样本对
2. 模型对所有样本进行前向传播计算
3. 计算每个标记的对数概率
4. 将结果分割为选中和拒绝两部分
5. 计算DPO损失函数

### 3.2 核心代码实现

DPO的前向传播函数：

```python
@override
def concatenated_forward(
    self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    计算给定logits下标签的对数概率总和（对于非IPO、ORPO或SimPO损失类型）
    否则计算平均对数概率。
    """
    if self.finetuning_args.use_ref_model:
        batch = nested_detach(batch, clone=True)  # 避免错误

    # 模型前向传播
    all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
    # 计算对数概率和有效长度
    all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
    if self.loss_type in ["ipo", "orpo", "simpo"]:
        all_logps = all_logps / valid_length

    # 分割为选中和拒绝的对数概率
    batch_size = batch["input_ids"].size(0) // 2
    chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
    chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
    chosen_length, _ = valid_length.split(batch_size, dim=0)

    if self.loss_type in ["ipo", "orpo", "simpo"]:
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps
    else:
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length
```

DPO批次损失计算：

```python
@override
def get_batch_loss_metrics(
    self,
    model: "PreTrainedModel",
    batch: Dict[str, "torch.Tensor"],
    train_eval: Literal["train", "eval"] = "train",
) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
    """
    计算给定输入批次的DPO损失和其他指标
    """
    metrics = {}
    # 进行前向传播，获取选中和拒绝样本的对数概率和logits
    (
        policy_chosen_logps,
        policy_rejected_logps,
        policy_chosen_logits,
        policy_rejected_logits,
        policy_chosen_logps_avg,
    ) = self.concatenated_forward(model, batch)

    # 获取参考模型的对数概率（如果使用参考模型）
    reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
    # 计算偏好损失
    losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    # 计算SFT损失部分
    sft_loss = -policy_chosen_logps_avg
    if self.ftx_gamma > 1e-6:
        losses += self.ftx_gamma * sft_loss

    # 记录训练指标
    prefix = "eval_" if train_eval == "eval" else ""
    metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
    metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
    metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
    metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
    metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
    metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
    metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
    metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
    if self.loss_type == "orpo":
        metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
        metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()

    return losses.mean(), metrics
```

## 4. 损失函数计算

DPO提供了多种损失函数实现，包括：

### 4.1 标准DPO损失

标准DPO损失从父类`DPOTrainer`继承，实现如下：

```python
def dpo_loss(
    self, 
    policy_chosen_logps, 
    policy_rejected_logps,
    reference_chosen_logps, 
    reference_rejected_logps
):
    # 计算策略模型与参考模型的对数概率之差
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    
    # 计算KL散度项
    logits = pi_logratios - ref_logratios
    
    # 应用标签平滑（可选）
    if self.label_smoothing > 0:
        targets = 1 - self.label_smoothing
        loss = (
            -F.logsigmoid(self.beta * logits) * targets
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )
    else:
        # 标准的DPO损失
        loss = -F.logsigmoid(self.beta * logits)
    
    # 计算奖励值（用于监控）
    chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
    
    return loss, chosen_rewards, rejected_rewards
```

### 4.2 Odds Ratio损失 (ORPO)

`CustomDPOTrainer`还实现了ORPO（赔率比）损失函数：

```python
def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
    """
    计算ORPO的赔率比损失，用于策略模型的批次对数概率
    """
    log_odds = (chosen_logps - rejected_logps) - (
        torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
    )
    sft_loss = -chosen_logps
    odds_ratio_loss = -F.logsigmoid(log_odds)
    orpo_loss = sft_loss + self.beta * odds_ratio_loss
    return orpo_loss
```

### 4.3 SimPO损失

此外，还实现了更简单的SimPO损失：

```python
def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
    """
    计算SimPO损失
    """
    pi_logratios = chosen_logps - rejected_logps
    gamma_logratios = self.simpo_gamma / self.beta
    logits = pi_logratios - gamma_logratios
    simpo_loss = -F.logsigmoid(self.beta * logits)
    return simpo_loss
```

### 4.4 偏好损失综合函数

`compute_preference_loss`函数根据配置选择使用哪种损失函数：

```python
def compute_preference_loss(
    self,
    policy_chosen_logps: "torch.Tensor",
    policy_rejected_logps: "torch.Tensor",
    reference_chosen_logps: Optional["torch.Tensor"],
    reference_rejected_logps: Optional["torch.Tensor"],
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    计算偏好学习的损失
    """
    if not self.finetuning_args.use_ref_model:
        if self.loss_type == "orpo":
            losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
        elif self.loss_type == "simpo":
            losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
        else:
            raise NotImplementedError(f"未知的损失类型: {self.loss_type}.")

        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
    else:
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
        )

    return losses, chosen_rewards, rejected_rewards
```

## 5. 与其他算法的区别

### 5.1 与SFT的区别

- **数据格式**：DPO使用偏好对（选中/拒绝）而不是单一的回复
- **损失函数**：DPO优化偏好损失，而SFT使用简单的语言模型负对数似然
- **参考模型**：DPO通常需要一个参考模型（通常是SFT训练的结果）
- **优化目标**：DPO旨在最大化模型生成选中回复的概率并最小化拒绝回复的概率

### 5.2 与PPO的区别

- **简化流程**：DPO避免了明确的奖励模型和复杂的策略优化步骤
- **训练稳定性**：DPO通常比PPO更稳定，不需要调整的超参数更少
- **计算效率**：DPO训练通常比PPO更高效，因为它避免了多次采样和策略更新
- **隐式奖励学习**：DPO隐式地学习奖励和策略，而不是显式建模

### 5.3 与KTO的区别

- **约束方式**：KTO（KL约束偏好优化）明确地约束策略更新与参考模型的KL散度
- **训练稳定性**：KTO通过KL约束为模型更新提供了更严格的控制，潜在地提高训练稳定性
- **损失函数**：KTO损失函数明确包含了KL散度项作为约束，而DPO将这种约束隐含在损失的计算中

## 6. 性能与实践考虑

- DPO通常在"人类对齐"任务中表现良好，可以有效提高模型输出的质量和有用性
- DPO的效果很大程度上依赖于偏好数据的质量和多样性
- 在训练过程中，beta参数（偏好损失的温度参数）是一个关键的超参数
- 对于更复杂的对齐任务，可能需要考虑联合使用DPO和SFT损失（通过ftx_gamma参数控制） 