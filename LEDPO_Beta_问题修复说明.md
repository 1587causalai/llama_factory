# LEDPO Beta趋零问题修复方案

## 问题背景

在LearnableBetaDPO(LEDPO)算法中，我们遇到了一个严重的问题：在训练过程中，`beta`值会不断减小并趋近于零，这与理论期望的行为相悖。

根据理论分析，`beta(x)`的行为应该是：
- 当`delta > 0`（模型正确偏好）时，`beta(x)`应该增大
- 当`delta < 0`（模型错误偏好）时，`beta(x)`应该减小

但实际训练中，无论`delta`值正负如何，`beta`值都在持续减小，最终趋近于零，导致模型无法正确学习偏好。

## 问题原因分析

经过深入分析代码和实验，我们发现了问题的主要原因：

1. **策略模型冻结问题**：当设置`freeze_policy_model=True`时，模型梯度无法正常传递到value_head，导致`beta_scale`参数的更新不正确。

2. **参数初始化与优化问题**：`beta_scale`参数的初始值和优化方式不适合当前任务，容易导致其值迅速衰减。

3. **梯度消失问题**：当`beta`值变小后，计算的梯度也变小，形成了负反馈循环，加速了`beta`值趋零的过程。

## 解决方案

我们提出了多方面的修复方案：

### 1. 修改ValueHead.forward方法

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # 输入是最后一层hidden states
    raw_beta = self.value_head(hidden_states)
    
    # 应用softplus确保beta_scale始终为正值
    beta_scale_positive = F.softplus(self.beta_scale)
    # 使用beta_scale_positive作为beta值的缩放因子
    scaled_beta = beta_scale_positive * raw_beta
    
    # 将输出截断到[beta_min, beta_max]范围
    clamped_beta = torch.clamp(scaled_beta, min=self.beta_min, max=self.beta_max)
    
    # 打印调试信息
    if torch.rand(1).item() < 0.1:  # 增加打印概率到10%
        logger.info(f"ValueHead forward - raw_beta: min={raw_beta.min().item():.4f}, max={raw_beta.max().item():.4f}, mean={raw_beta.mean().item():.4f}")
        logger.info(f"ValueHead forward - beta_scale (raw): {self.beta_scale.item():.4f}")
        logger.info(f"ValueHead forward - beta_scale (positive): {beta_scale_positive.item():.4f}")
        logger.info(f"ValueHead forward - scaled_beta: min={scaled_beta.min().item():.4f}, max={scaled_beta.max().item():.4f}, mean={scaled_beta.mean().item():.4f}")
    
    return clamped_beta
```

关键修改：
- 使用`F.softplus`函数确保`beta_scale`始终为正值
- 增加调试信息，便于监控`beta`值的变化

### 2. 添加beta_scale正则化

在`compute_preference_loss`方法中添加正则化项：

```python
# 防止使用freeze_policy_model时beta_scale不断减小
if self.freeze_policy_model and hasattr(self.value_head, "beta_scale"):
    # 计算正则化损失: 鼓励beta_scale不要过小，保持在合理范围内
    beta_scale_threshold = 1.0
    if self.value_head.beta_scale.item() < beta_scale_threshold:
        beta_reg_factor = 0.1  # 正则化系数
        beta_reg_loss = beta_reg_factor * F.relu(beta_scale_threshold - self.value_head.beta_scale)
        logger.info(f"Adding beta_scale regularization: {beta_reg_loss.item():.4f}")
        
# 添加正则化损失到总损失
if beta_reg_loss.item() > 0:
    losses = losses + beta_reg_loss
```

关键修改：
- 当`beta_scale`低于阈值时添加正则化损失，防止其持续减小
- 将正则化损失添加到总损失中

### 3. 优化器参数调整

在`create_optimizer`方法中进行调整：

```python
# 为value_head参数设置更高的学习率
value_head_lr = self.args.learning_rate * 20.0  # 增加到原来的20倍

# 添加参数组
params_config = {
    "params": value_head_params,
    "lr": value_head_lr,  # 使用更高的学习率
    "weight_decay": 0.0,  # 减少权重衰减，防止beta_scale变小
}

# 确保beta_scale有足够大的初始值
with torch.no_grad():
    if self.value_head.beta_scale.item() < 5.0:
        self.value_head.beta_scale.fill_(10.0)
```

关键修改：
- 为`value_head`参数设置更高的学习率，加速其学习
- 减少权重衰减，防止`beta_scale`变小
- 确保`beta_scale`保持较大的初始值

### 4. 配置参数调整

在配置文件中进行调整：

```yaml
use_dynamic_beta: true  # 启用动态beta
beta_min: 0.1  # 动态beta的最小值 - 增加最小值，防止beta过小
beta_max: 50.0  # 动态beta的最大值 - 适当减小最大值
freeze_policy_model: false  # 不冻结策略模型，让梯度能正常传递
learning_rate: 5.0e-4  # 增加学习率
```

关键修改：
- 将`freeze_policy_model`设为`false`，解决主要问题
- 增加`beta_min`的值，防止beta过小
- 调整学习率，加速模型学习

## 实施步骤

1. 备份原始文件:
   ```bash
   cp src/llamafactory/train/ledpo/trainer.py src/llamafactory/train/ledpo/trainer.py.bak
   ```

2. 修改ValueHead.forward方法
3. 修改compute_preference_loss方法
4. 修改create_optimizer方法
5. 使用新的配置文件进行训练

可以使用提供的`run_ledpo_fixed.sh`脚本来简化上述过程。

## 效果验证

修复后，我们预期会观察到：
1. `beta_scale`值保持稳定或根据数据分布合理变化
2. `delta > 0`时的平均beta值高于`delta < 0`时的平均beta值
3. 模型能够更好地学习偏好，训练loss正常下降

## 注意事项

1. 如果不想修改原始代码，可以考虑关闭`freeze_policy_model`选项（这是最简单的修复方法）
2. 适当增加`beta_min`值有助于避免beta过小的问题
3. 使用cosine学习率调度器和热身期可以使训练更稳定

## 总结

LearnableBetaDPO算法中beta趋零问题主要由`freeze_policy_model`选项引起，通过修改代码实现和调整配置参数，我们可以解决这个问题，使beta值按照理论预期变化，从而提高模型学习人类偏好的效果。 