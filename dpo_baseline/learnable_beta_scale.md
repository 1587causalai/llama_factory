# BetaDPO可学习Beta_Scale实现文档

## 一、背景介绍

BetaDPO是在标准DPO基础上的改进算法，核心创新在于使用**动态beta机制**来调整偏好学习强度。在原始方案中，beta值是根据文本困惑度（Perplexity）动态调整的：

```
动态beta = beta_scale × log(PPL(x)) × beta基准值
```

其中：
- `beta基准值`：基础beta参数，通常设置为0.1-0.2
- `log(PPL(x))`：输入文本的对数困惑度，反映文本复杂度
- `beta_scale`：控制动态调整强度的缩放系数

传统实现中，`beta_scale`是一个固定的超参数，需要手动调整。我们提出将`beta_scale`改为可学习参数，让模型根据训练信号自动找到最优值。

## 二、可学习Beta_Scale的实现

### 1. 核心实现思路

将`beta_scale`从固定超参数转变为**可训练模型参数**的关键步骤：

1. 将`beta_scale`初始化为`torch.nn.Parameter`并添加到模型中
2. 在优化器中为`beta_scale`设置单独的参数组
3. 在损失计算过程中使用可学习的`beta_scale`
4. 在训练和评估过程中记录`beta_scale`的变化

### 2. 代码实现详解

#### 2.1 参数初始化

在`CustomBetaDPOTrainer`初始化中，创建可学习参数并添加到模型：

```python
# 在__init__方法中
beta_scale_param = torch.nn.Parameter(torch.tensor(finetuning_args.pref_beta_scale, dtype=torch.float32))

# 调用Trainer初始化
Trainer.__init__(self, model=model, **kwargs)

# 将beta_scale设置为模型的属性（在Trainer初始化后）
self.model.beta_scale = beta_scale_param
```

#### 2.2 优化器配置

重写`create_optimizer`方法，单独处理`beta_scale`参数：

```python
@override
def create_optimizer(self) -> "torch.optim.Optimizer":
    if self.optimizer is None:
        # 检查模型是否有beta_scale属性
        if not hasattr(self.model, "beta_scale"):
            # 如果没有beta_scale属性，使用原始方法创建优化器
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
            return super().create_optimizer()
        
        # 根据不同的优化器类型处理
        if hasattr(self.finetuning_args, "use_galore") and self.finetuning_args.use_galore:
            # GaLore优化器特殊处理
            from ..trainer_utils import _create_galore_optimizer
            self.optimizer = _create_galore_optimizer(self.model, self.args, self.finetuning_args)
            # 添加beta_scale参数组
            self.optimizer.add_param_group({"params": [self.model.beta_scale], "weight_decay": 0.0})
        elif hasattr(self.finetuning_args, "use_apollo") and self.finetuning_args.use_apollo:
            # Apollo优化器特殊处理
            from ..trainer_utils import _create_apollo_optimizer
            self.optimizer = _create_apollo_optimizer(self.model, self.args, self.finetuning_args)
            # 添加beta_scale参数组
            self.optimizer.add_param_group({"params": [self.model.beta_scale], "weight_decay": 0.0})
        else:
            # 标准优化器创建
            optimizer_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            
            # 创建两个参数组：一个用于模型参数，一个用于beta_scale
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() 
                              if p.requires_grad and n != "beta_scale"],
                },
                {
                    "params": [self.model.beta_scale],
                    "weight_decay": 0.0,  # 不对beta_scale应用权重衰减
                }
            ]
            
            # 创建包含所有参数组的优化器
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters,
                **optim_kwargs,
            )
    
    return super().create_optimizer()
```

#### 2.3 损失计算中使用可学习参数

在`compute_preference_loss`方法中获取并使用可学习的`beta_scale`：

```python
def compute_preference_loss(
    self,
    policy_chosen_logps: "torch.Tensor",
    policy_rejected_logps: "torch.Tensor",
    reference_chosen_logps: Optional["torch.Tensor"],
    reference_rejected_logps: Optional["torch.Tensor"],
    perplexity: "torch.Tensor" = None,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    计算偏好学习的损失，使用动态beta
    """
    # 计算动态beta: β(x) = c · log(PPL(x)) · β
    base_beta = self.beta
    
    # 从模型获取beta_scale参数
    beta_scale_param = self.model.beta_scale
    
    if perplexity is not None:
        # 增加数值稳定性
        log_ppl = torch.log(torch.clamp(perplexity, min=1.0))
        
        # 使用softplus确保beta_scale为正值
        beta_scale_value = F.softplus(beta_scale_param)
        
        # 使用可学习的beta_scale计算动态beta
        dynamic_beta = beta_scale_value * log_ppl * base_beta # dynamic_beta.shape = [batch_size]
    else: # 当perplexity为None时
        # 使用可学习的beta_scale
        beta_scale_value = F.softplus(beta_scale_param)
        dynamic_beta = base_beta * torch.ones_like(policy_chosen_logps) * beta_scale_value
    
    # 使用动态beta替代固定beta，其余逻辑与原实现相同
    if not self.finetuning_args.use_ref_model:
        if self.loss_type == "orpo":
            losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps, dynamic_beta)
        elif self.loss_type == "simpo":
            losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps, dynamic_beta)
        else:
            # 标准DPO损失
            losses, chosen_rewards, rejected_rewards = self.dpo_loss_with_dynamic_beta(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                dynamic_beta,
            )
    else:
        # 使用参考模型的情况
        losses, chosen_rewards, rejected_rewards = self.dpo_loss_with_dynamic_beta(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            dynamic_beta,
        )
        
    return losses, chosen_rewards, rejected_rewards
```

#### 2.4 日志记录

重写`log`方法，添加对`beta_scale`值的记录：

```python
@override
def log(self, logs: Dict[str, float], *args, **kwargs) -> Dict[str, float]:
    # 其他日志处理...
    
    # 记录beta_scale的当前值（从模型中获取）
    if hasattr(self.model, "beta_scale"):
        logs[f"{train_eval}_beta_scale"] = F.softplus(self.model.beta_scale).item()

    return Trainer.log(self, logs, *args, **kwargs)
```

### 3. 动态beta的使用

我们已经确保了以下几个关键函数支持动态beta：

1. `odds_ratio_loss`: 支持传入`dynamic_beta`参数计算ORPO损失
2. `simpo_loss`: 支持传入`dynamic_beta`参数计算SimPO损失 
3. `dpo_loss_with_dynamic_beta`: 专门处理动态beta的DPO损失计算

这些方法共同确保了BetaDPO算法在各种损失函数下都能正确使用可学习的beta_scale参数。

## 三、配置与使用方法

### 1. 配置文件示例

```yaml
### method
stage: betadpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
pref_beta: 0.1       # 基准beta值
pref_loss: sigmoid   # 损失类型：sigmoid (dpo), orpo, simpo
pref_beta_scale: 0.5 # beta_scale初始值

### 训练参数
learning_rate: 1.0e-4
```

### 2. 使用方法

1. 在配置文件中设置`pref_beta_scale`初始值（推荐范围0.1-1.0）
2. 使用BetaDPO训练命令启动训练
3. 在训练日志中监控`train_beta_scale`和`eval_beta_scale`值的变化
4. 训练完成后，学习到的beta_scale会自动保存为模型的一部分

```bash
# 运行命令示例
llamafactory-cli train examples/train_lora/qwen1_5_0_5b_lora_betadpo.yaml
# 或使用详细脚本
python dpo_baseline/run_betadpo_detailed.py examples/train_lora/qwen1_5_0_5b_lora_betadpo.yaml
```

## 四、设计优势

### 1. 自动化调整

将beta_scale作为可学习参数的主要优势：

- **消除超参数调优**：不再需要手动寻找最佳beta_scale值
- **自适应性能**：模型能够根据数据特点自动调整beta_scale
- **学习曲线平滑**：避免人工设置不当导致的训练不稳定

### 2. 实现优势

- **最小化修改**：对现有代码的改动极小，容易整合到其他实现中
- **统一参数管理**：作为模型参数的一部分，自动被保存和加载
- **框架兼容性**：完全兼容HuggingFace Trainer框架
- **优化灵活性**：支持为beta_scale设置独立的优化参数

### 3. 兼容性和稳定性

- **数值稳定性**：使用softplus函数确保beta_scale始终为正值
- **适用多种损失**：支持标准DPO、ORPO和SimPO等多种偏好学习损失函数
- **优化器兼容**：支持常规优化器、GaLore、Apollo等高级优化器
- **回退机制**：当未设置beta_scale时自动回退到固定beta行为

## 五、实验结果与发现

(实验部分将在完成训练后补充，包括：)

1. 不同初始值下beta_scale的收敛趋势
2. 学习到的beta_scale值与手动调优值的比较
3. 对不同数据集和模型规模的适应性
4. 对训练稳定性和性能的影响

## 六、结论与未来工作

将beta_scale从固定超参数改为可学习参数是对BetaDPO算法的重要改进，它使算法能够自动适应不同的数据分布和模型规模，减少了人工调参的需求，同时提高了训练效果和稳定性。

未来工作方向：
1. 探索beta_scale与其他超参数的联合学习
2. 研究不同初始化策略对学习速度的影响 
3. 设计更复杂的beta调整机制，如引入上下文相关的动态调整
4. 研究beta_scale与模型规模的关系，为不同规模模型推荐最佳初始值 