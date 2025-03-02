# KTO (Kahneman-Tversky Optimization) 前向传播分析

## 1. 算法原理概述

KTO (Kahneman-Tversky Optimization) 是一种基于行为经济学前景理论（Prospect Theory）的模型对齐方法，由Kawin Ethayarajh等人在2024年提出。前景理论是由Kahneman和Tversky创立的，描述了人类在面对不确定性时如何做出决策，特别是人类对收益和损失的非对称感知（人类通常更关注损失而非收益）。

KTO的核心创新点在于：
- 直接使用Kahneman-Tversky模型的人类效用函数来优化模型输出
- 只需要一个二元信号（内容是否可取），而不需要完整的偏好对
- 显式建模人类对收益和损失的非对称感知

与DPO等方法不同，KTO并非对偏好对数据的对数似然进行优化，而是直接最大化生成内容的人类感知效用。

数学细节见: https://grok.com/share/bGVnYWN5_d9fdad38-2fa6-4d64-9bb4-bd94f7d6745f

## 2. 数据流和输入格式

### 2.1 输入数据格式

KTO的数据输入可以分为两类：

1. **可取内容（Desirable Contents）**：被标记为高质量或符合要求的回复
2. **不可取内容（Undesirable Contents）**：被标记为低质量或不符合要求的回复

每个样本包含以下组成部分：
- **提示（Prompt）**：用户输入或任务描述
- **回复（Response）**：模型生成的回复
- **标签（Tag）**：二元标签，表示该回复是可取的还是不可取的

与DPO不同，KTO不要求成对的偏好数据，只需要对每个输出的二元评价。

### 2.2 数据批次组织

在LLaMA Factory的实现中，KTO使用`kto_tags`来标识批次中的可取和不可取样本：

```python
# 对样本进行标记
batch["kto_tags"] = torch.tensor([True, False, True, False, ...])  # 示例
```

此外，KTO还使用额外的KL样本来计算KL散度约束。

## 3. 前向传播实现

### 3.1 总体流程

KTO的前向传播流程主要由以下步骤组成：

1. 处理输入批次，识别可取和不可取样本
2. 对所有样本执行模型前向计算，获取logits和对数概率
3. 计算参考模型（reference model）的对数概率
4. 应用前景理论模型计算KTO损失

### 3.2 核心代码实现

KTO的基本前向传播实现：

```python
def forward(self, model, batch, prefix=""):
    """运行前向传播并计算对数概率"""
    batch = nested_detach(batch, clone=True)
    model_inputs = {
        "input_ids": batch[f"{prefix}input_ids"],
        "attention_mask": batch[f"{prefix}attention_mask"],
    }
    # 处理其他输入
    if f"{prefix}token_type_ids" in batch:
        model_inputs["token_type_ids"] = batch[f"{prefix}token_type_ids"]
    # 多模态输入处理
    if "pixel_values" in batch:
        model_inputs["pixel_values"] = batch["pixel_values"]
    # ...其他输入处理...

    # 模型前向计算
    logits = model(**model_inputs, return_dict=True, use_cache=False).logits.to(torch.float32)
    logps, valid_length = get_batch_logps(logits=logits, labels=batch[f"{prefix}labels"])
    return logits, logps, logps / valid_length
```

连接前向传播函数，区分可取和不可取样本：

```python
def concatenated_forward(self, model, batch):
    """执行前向计算并区分可取和不可取样本"""
    # 标准前向传播
    target_logits, target_logps, target_logps_avg = self.forward(model, batch)
    # 计算KL样本的对数概率
    with torch.no_grad():
        _, kl_logps, _ = self.forward(model, batch, prefix="kl_")

    # 使用KTO标签分离可取和不可取样本
    chosen_logits = target_logits[batch["kto_tags"]]
    chosen_logps = target_logps[batch["kto_tags"]]
    rejected_logits = target_logits[~batch["kto_tags"]]
    rejected_logps = target_logps[~batch["kto_tags"]]
    chosen_logps_avg = target_logps_avg[batch["kto_tags"]]
    
    return chosen_logps, rejected_logps, chosen_logits, rejected_logits, kl_logps, chosen_logps_avg
```

批次损失计算：

```python
def get_batch_loss_metrics(self, model, batch):
    """计算KTO损失和训练指标"""
    metrics = {}
    # 执行前向传播
    (
        policy_chosen_logps,
        policy_rejected_logps,
        policy_chosen_logits,
        policy_rejected_logits,
        policy_kl_logps,
        policy_chosen_logps_avg,
    ) = self.concatenated_forward(model, batch)
    
    # 获取参考模型的对数概率
    reference_chosen_logps, reference_rejected_logps, reference_kl_logps = self.compute_reference_log_probs(
        model, batch
    )
    
    # 计算KTO损失
    losses, chosen_rewards, rejected_rewards, kl = self.kto_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        policy_kl_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        reference_kl_logps,
    )
    losses = losses.nanmean()

    # 可选：混合SFT损失
    if self.ftx_gamma > 1e-6 and len(policy_chosen_logps) > 0:
        sft_loss = -policy_chosen_logps_avg
        losses += self.ftx_gamma * sft_loss.nanmean() / len(policy_chosen_logps) * len(batch["labels"])

    # 记录各种训练指标
    # ...

    metrics["kl"] = kl.item()
    return losses, metrics
```

## 4. 损失函数计算

KTO的核心在于其基于前景理论的损失函数设计。

### 4.1 前景理论与KTO损失

前景理论模型包含两个关键组件：
1. **价值函数(Value Function)**：描述收益和损失对人类感知的非对称影响
2. **决策权重(Decision Weights)**：描述人类对概率的非线性感知

KTO损失函数的设计直接反映了这些特性：

```python
def kto_loss(
    self,
    policy_chosen_logps,
    policy_rejected_logps,
    policy_kl_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    reference_kl_logps,
):
    # 计算日志概率差异
    chosen_kl = (policy_chosen_logps - reference_chosen_logps)
    rejected_kl = (policy_rejected_logps - reference_rejected_logps)
    
    # 应用前景理论中的权重
    # desirable_weight对应可取内容的权重
    # undesirable_weight对应不可取内容的权重
    preference_loss = (
        -self.desirable_weight * chosen_kl
        + self.undesirable_weight * rejected_kl
    )
    
    # KL散度约束
    kl_divergence = (policy_kl_logps - reference_kl_logps).mean()
    
    # 组合损失
    loss = preference_loss - self.beta * kl_divergence
    
    # 计算奖励（用于监控）
    chosen_rewards = self.beta * chosen_kl.detach()
    rejected_rewards = self.beta * rejected_kl.detach()
    
    return loss, chosen_rewards, rejected_rewards, kl_divergence
```

### 4.2 KTO特有参数

KTO引入了两个关键参数来实现前景理论中的非对称效应：

- **desirable_weight**：控制对可取内容的重视程度
- **undesirable_weight**：控制对不可取内容的重视程度

这两个参数的不对称设置反映了前景理论中损失规避（loss aversion）的核心思想：人类对损失的感知通常比对等量收益的感知更强烈。在实践中，通常设置`undesirable_weight > desirable_weight`。

### 4.3 KTO损失函数的数学公式

根据KTO论文和代码实现，我们可以详细推导KTO损失函数的数学表达式。KTO损失函数基于前景理论中的价值函数和决策权重函数，通过直接优化人类感知效用来实现模型对齐。

My comment: https://grok.com/share/bGVnYWN5_2fde0250-38a9-49f4-ba82-39655389acec

#### 前景理论背景

前景理论中，人类对收益和损失的评估通过价值函数(Value Function) $v(x)$ 表示:

$$v(x) = \begin{cases}
x^\alpha, & \text{if } x \geq 0 \\
-\lambda(-x)^\beta, & \text{if } x < 0
\end{cases}$$

其中:
- $x$ 是收益或损失
- $\alpha, \beta$ 是风险敏感度参数（通常 $\alpha, \beta < 1$，表示风险规避）
- $\lambda > 1$ 是损失规避系数，反映损失对人类的影响比等量收益更大

决策权重函数 $\pi(p)$ 将客观概率 $p$ 映射为主观决策权重:

$$\pi(p) = \frac{p^\gamma}{(p^\gamma + (1-p)^\gamma)^{1/\gamma}}$$

其中 $\gamma$ 控制概率权重曲线的形状。

#### KTO损失函数推导

在KTO中，这些概念被应用于语言模型训练，通过以下方式:

1. **定义效用差异**
   
   对于可取内容和不可取内容，我们计算策略模型与参考模型之间的对数概率差异:
   
   $$\Delta_{+} = \log\pi_\theta(y^+|x) - \log\pi_{\text{ref}}(y^+|x)$$
   $$\Delta_{-} = \log\pi_\theta(y^-|x) - \log\pi_{\text{ref}}(y^-|x)$$
   
   其中:
   - $\pi_\theta$ 是当前策略模型
   - $\pi_{\text{ref}}$ 是参考模型
   - $y^+$ 是可取内容，$y^-$ 是不可取内容
   - $x$ 是输入提示

2. **应用前景理论价值函数**
   
   KTO将前景理论的非对称价值感知应用于这些差异，简化后的形式为:
   
   $$L_{\text{preference}} = -w_{+} \cdot \Delta_{+} + w_{-} \cdot \Delta_{-}$$
   
   其中 $w_{+}$ (desirable_weight) 和 $w_{-}$ (undesirable_weight) 是权重参数，通常 $w_{-} > w_{+}$ 以反映损失规避特性。

3. **添加KL散度约束**
   
   为防止模型过度偏离参考模型，KTO添加KL散度约束:
   
   $$\text{KL}(\pi_\theta || \pi_{\text{ref}}) = \mathbb{E}_x[\log\pi_\theta(x) - \log\pi_{\text{ref}}(x)]$$

4. **完整的KTO损失函数**
   
   结合前面的组件，完整的KTO损失函数为:
   
   $$L_{\text{KTO}} = -w_{+} \cdot \Delta_{+} + w_{-} \cdot \Delta_{-} - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})$$
   
   其中 $\beta$ 是控制KL散度约束强度的参数。

5. **实现层面的数学表达**
   
   从代码实现看，KTO损失的具体计算为:
   
   $$L_{\text{KTO}} = -w_{+} \cdot (L_\theta^+ - L_{\text{ref}}^+) + w_{-} \cdot (L_\theta^- - L_{\text{ref}}^-) - \beta \cdot (L_\theta^{\text{KL}} - L_{\text{ref}}^{\text{KL}})$$
   
   其中:
   - $L_\theta^+, L_{\text{ref}}^+$ 是策略模型和参考模型对可取内容的对数概率
   - $L_\theta^-, L_{\text{ref}}^-$ 是对不可取内容的对数概率
   - $L_\theta^{\text{KL}}, L_{\text{ref}}^{\text{KL}}$ 是用于计算KL散度的样本对数概率

#### 与传统方法的数学比较

与标准DPO损失相比:

$$L_{\text{DPO}} = -\mathbb{E}_{(x,y^+,y^-)}[\log\sigma(\beta(r_\theta(x,y^+) - r_\theta(x,y^-)))]$$

其中 $r_\theta(x,y) = \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$

KTO更直接地建模了前景理论中的非对称效应，通过不同的权重处理可取和不可取内容。

而相比于PPO:

$$L_{\text{PPO}} = \mathbb{E}_\tau[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$

KTO避免了显式的奖励模型和多次策略迭代，同时仍然保持了类似的KL约束机制，但采用了更符合人类决策模式的非对称价值函数。

#### 超参数设置的数学解释

从前景理论角度来看，超参数的设置有明确的数学意义:

- $w_{+} < w_{-}$ 反映了损失规避 ($\lambda > 1$)
- $\beta$ 类似于PPO中的KL惩罚系数，但从人类决策的角度进行了调整
- 如果 $w_{+} = w_{-}$，KTO将退化为类似于对称的偏好优化方法

通过这种方式，KTO将前景理论中的数学模型直接映射到语言模型训练中，形成了一套理论上更符合人类决策过程的对齐方法。

## 5. 与其他算法的区别

### 5.1 与DPO的本质区别

- **理论基础**：KTO基于前景理论，DPO基于Bradley-Terry模型
- **数据需求**：KTO只需要二元标签，DPO需要成对的偏好数据
- **优化目标**：KTO直接优化人类感知效用，DPO优化偏好对的对数似然
- **非对称处理**：KTO显式建模收益和损失的非对称性，DPO对两种情况的处理是对称的
- **KL约束**：KTO显式包含KL约束，DPO将KL约束隐含在损失函数中

### 5.2 与SFT的区别

- **学习目标**：KTO学习人类偏好，SFT学习模仿训练数据
- **数据格式**：KTO使用带标签的回复，SFT使用指令-回复对
- **优化方法**：KTO基于前景理论优化效用，SFT基于最大似然估计
- **参考模型**：KTO需要一个参考模型，通常是SFT训练的结果

### 5.3 与PPO的区别

- **理论基础**：KTO基于前景理论，PPO基于策略梯度和信任区域优化
- **训练流程**：KTO避免了显式的奖励模型，PPO需要奖励模型和多次策略更新
- **计算复杂度**：KTO训练效率更高，PPO需要多次采样和评估
- **稳定性**：KTO通常比PPO更稳定，对超参数选择不那么敏感

## 6. 性能与实践考虑

- **数据效率**：KTO只需要二元标签而非完整偏好对，在某些场景下数据效率更高
- **训练稳定性**：KTO的显式KL约束和前景理论模型提供了更稳定的训练过程
- **超参数敏感度**：KTO对beta参数的敏感度较低，但需要适当设置desirable_weight和undesirable_weight
- **适用场景**：KTO特别适合于需要非对称处理可取和不可取行为的对齐任务
- **扩展性**：KTO框架可以进一步扩展，纳入更复杂的人类决策模型

总体而言，KTO通过引入行为经济学的前景理论，为语言模型对齐提供了一种理论上更符合人类决策过程的方法。它不仅简化了训练流程，还提供了控制模型行为非对称性的灵活机制。 