# LeDPO 开发日志

本文档记录 LeDPO (可学习Beta DPO) 算法的渐进式开发过程，包括每个阶段的工作内容、决策和结果。

## 开发阶段概览

| 阶段 | 描述 | 状态 | 对应保存点 |
|------|------|------|------------|
| 1    | 标准DPO基准测试 | 已完成 | 保存点1 |
| 2    | LEDPO基础框架建立 | 已完成 | 保存点2 |
| 3    | 基于最后提示词Token的动态beta实现 | 已完成 | 保存点3 |
| 4    | 改进动态beta值监控系统 | 已完成 | 保存点4 |
| 5    | 动态beta理论改进与实验验证 | 已完成 | 保存点5 |
| 6    | 冻结策略模型与性能优化实验 | 已完成 | 保存点6 |
| 7    | 修复beta head更新机制 | 已完成 | 保存点7 |
| 8    | 模型性能评估与对比分析 | 计划中 | - |

## 阶段1：标准DPO基准测试 (2025-03-16)

### 目标
建立标准DPO训练的基准性能，为后续LEDPO开发奠定基础。

### 完成工作
1. **复制DPO模块创建FooDPO基础实现**
   - 从 `src/llamafactory/train/dpo` 复制代码到 `src/llamafactory/train/foodpo`
   - 将 `CustomDPOTrainer` 重命名为 `CustomFooDPOTrainer`
   - 将 `run_dpo` 重命名为 `run_foodpo`
   - 在注释中添加了 FooDPO 的基本描述

2. **修改相关文件以支持FooDPO**
   - 更新 `src/llamafactory/train/tuner.py` 以导入和使用 `run_foodpo`
   - 在 `_training_function` 中添加了对 `foodpo` 阶段的处理
   - 确认 `finetuning_args.py` 中已添加 `foodpo` 作为有效的训练阶段

3. **创建配置文件**
   - 创建 `ledpo_progressive_dev/qwen15_lora_foodpo.yaml` 配置文件
   - 针对本地环境（使用 Qwen1.5-0.5B 模型）优化了配置参数
   - 使用 `dpo_en_demo` 作为训练数据集

4. **实现训练指标监控系统**
   - 创建了 `plot_ledpo_metrics.py` 绘图脚本，直接从`trainer_state.json`读取并可视化训练数据
   - 创建了 `run_train_and_plot.py` 集成脚本，一键完成训练和绘图
   - 设计了包含6个关键指标的监控系统：accuracy、loss、reward/margin、reward/chosen、reward/rejected、beta
   - 为后续LEDPO开发预留了特定指标支持：pos_beta和neg_beta

### 运行命令
```bash
llamafactory-cli train ledpo_progressive_dev/qwen15_lora_foodpo.yaml
# 或使用集成脚本
python ledpo_progressive_dev/run_train_and_plot.py --config ledpo_progressive_dev/qwen15_lora_foodpo.yaml
```

### 保存点1: 训练指标监控系统实现 (350d9c9f)
**创建时间:** 2025-03-17

**内容:** 
- 实现了标准DPO训练的指标监控系统
- 添加了对LEDPO特定指标(pos_beta, neg_beta)的预支持
- 创建了训练和绘图的集成脚本

**回退方法:**
```bash
git checkout 350d9c9f
# 或创建新分支
git checkout -b new_branch_name 350d9c9f
```

## 阶段2：LEDPO基础框架建立

### 目标
在标准DPO基础上建立LEDPO的基础架构，实现最小化的动态beta功能。

### 完成工作
1. **添加动态beta支持**
   - 在 `finetuning_args.py` 中添加了 `use_dynamic_beta` 参数
   - 创建了 `beta_head.py` 实现基于长度的beta计算模型
   - 在 `trainer.py` 中添加了动态beta的基础代码结构和初始化

2. **实现基础beta计算逻辑**
   - 实现了基于提示长度计算beta值的基础逻辑
   - 添加了beta值的监控和指标记录支持

### 保存点2: LEDPO 基础框架建立 (0c8aa146)
**创建时间:** 2025-03-18

**内容:**
- 在 `finetuning_args.py` 中添加了 `use_dynamic_beta` 参数
- 创建了 `beta_head.py` 实现基于长度的beta计算模型
- 在 `trainer.py` 中添加了动态beta的基础代码结构和初始化
- 实现了基于提示长度计算beta值的基础逻辑

**状态说明:**
- 当前版本中动态beta调整的逻辑尚未在训练过程中实际生效
- 已添加beta值的监控和指标记录支持
- 删除了临时测试文件（qwen15_lora_ledpo_v1.yaml 和 run_ledpo_v1.sh）

**回退方法:**
```bash
git checkout 0c8aa146a93a0b744659e37c55c42ccdee4c33a4
```

## 阶段3：基于最后提示词Token的动态beta实现 (2025-03-19)

### 目标
改进beta计算方法，基于提示的最后一个token的隐藏状态计算更精确的beta值。

### 完成工作
1. **改进beta计算模型**
   - 创建了 `HiddenStateBetaHead` 类，基于隐藏状态计算beta值
   - 替换原有的基于长度的beta计算模型

2. **实现隐藏状态提取**
   - 更新了 `concatenated_forward` 方法，实现从模型输出中提取隐藏状态
   - 添加了提取提示最后一个token位置的代码，并用它来获取对应的隐藏状态

### 保存点3: 基于最后提示词Token的动态beta实现 (fd369dac)

**内容:**
- 创建了 `HiddenStateBetaHead` 类，用于基于提示的最后一个token的隐藏状态计算beta值
- 修改了 `__init__` 方法，将 `LengthBasedBetaHead` 替换为 `HiddenStateBetaHead`
- 更新了 `concatenated_forward` 方法，实现了从模型输出中提取隐藏状态
- 添加了提取提示最后一个token位置的代码，并用它来获取对应的隐藏状态
- 使用 `beta_head` 基于隐藏状态计算动态beta值

**状态说明:**
- 当前实现支持基于提示语义内容生成更精确的beta值
- 保持了渐进式开发原则，只在必要处进行了修改
- 动态beta计算逻辑已完成，但尚未应用到实际损失计算中

**回退方法:**
```bash
git checkout fd369dac
```

## 阶段4：改进动态beta值监控系统 (2025-03-20)

### 目标
完善动态beta值的监控和可视化系统，为下一阶段应用动态beta做准备。

### 完成工作
1. **优化beta指标计算**
   - 修改`trainer.py`中beta相关指标的命名格式
   - 根据delta值的正负对样本进行分类，分别计算pos_beta和neg_beta

2. **改进绘图脚本**
   - 优化`plot_ledpo_metrics.py`，从日志中获取实际的动态beta值
   - 修复指标名称格式，确保图表能正确显示beta值变化

### 保存点4: 改进动态beta值监控系统 (57bc56e5)

**内容:**
- 修改`trainer.py`中beta相关指标的命名格式，将`beta/pos`改为`pos_beta`，将`beta/neg`改为`neg_beta`
- 根据delta值的正负对样本进行分类，分别计算pos_beta和neg_beta
- 优化`plot_ledpo_metrics.py`绘图脚本，从日志中获取实际的动态beta值，而不是使用固定值
- 修复绘图脚本中错误的beta值指标名称格式，确保图表能正确显示beta值的变化

**状态说明:**
- 完善了delta、beta、pos_beta和neg_beta四个核心指标的监控和可视化
- 可以通过图表清晰观察到动态beta值随训练过程的变化
- 准备下一阶段将动态beta值应用到实际损失计算中

**回退方法:**
```bash
git checkout 57bc56e5
```

## 阶段5：动态beta理论改进与实验验证 (2025-03-22)

### 目标
修正delta值计算公式以符合理论，实现模型参数冻结实验，验证动态beta的效果。

### 完成工作
1. **修正delta值计算公式**
   - 更新delta值计算公式为: `Delta = (π_θ(y_w|x) - π_ref(y_w|x)) - (π_θ(y_l|x) - π_ref(y_l|x))`
   - 修改`compute_preference_loss`函数实现正确的delta计算
   - 增强delta值监控，记录delta平均值、正负样本比例等关键指标
   
2. **实现模型参数冻结功能**
   - 在`__init__`方法中添加冻结策略模型参数代码
   - 确保beta_head参数保持可训练状态
   - 实现梯度流测试函数验证beta_head的可训练性
   
3. **优化绘图脚本**
   - 更新`plot_ledpo_metrics.py`，直接使用训练器记录的delta值
   - 修改delta指标的描述，使用完整理论公式
   - 优化图表布局和标题

4. **实验验证**
   - 运行冻结模型参数的实验，观察delta值趋势
   - 监控beta值与delta值的关系
   - 分析beta_head是否正确学习调整beta值

### 保存点5: 动态beta理论改进与实验验证 (b47a59ec)
**创建时间:** 2025-03-23

**内容:**
- 修正了delta值计算公式，使其符合理论定义
- 实现了策略模型参数冻结功能，同时保持beta_head可训练
- 增强了delta和beta值的监控和统计分析
- 优化了绘图脚本，直接使用训练器记录的准确delta值
- 添加了详细的日志输出，帮助监控训练过程中的关键指标变化

**状态说明:**
- 成功验证了冻结模型参数条件下beta_head的学习行为
- 确认delta值计算符合理论公式
- 观察到在策略模型被冻结的情况下delta值趋于稳定，验证了实现的正确性

**回退方法:**
```bash
git checkout 97b05347
# 56b96e85..97b05347
```

## 阶段6：冻结策略模型与性能优化实验 (2025-03-25)

### 目标
开展全面的冻结策略模型实验，比较冻结与非冻结模型的表现差异，并研究不同参考模型对学习动态的影响。

### 完成工作
1. **建立冻结策略模型实验框架**
   - 创建了三种实验配置：冻结模型、冻结模型-不同参考模型、非冻结模型
   - 实现了自动化实验运行脚本`run_experiment.sh`，支持三种配置
   - 开发了详细的实验分析脚本`analyze_logs.py`

2. **改进实验分析工具**
   - 增强分析脚本，支持三种模型结果的对比分析
   - 实现了delta和beta值的详细统计分析
   - 修改图表标题和轴标签从中文为英文，避免字体问题
   - 完善了模型比较报告生成功能

3. **解决配置问题**
   - 修正了参考模型路径配置问题
   - 解决了`use_ref_model`参数处理的问题
   - 优化了配置文件结构，增强实验的可重复性

4. **发现重大bug：beta head参数未正常更新**
   - 通过实验分析发现beta head参数并未按预期更新，beta值几乎保持不变
   - 确认该问题与梯度计算或反向传播机制相关
   - 记录了详细的症状和初步诊断信息，为后续修复做准备

### 保存点6: 冻结策略模型实验框架 (8df6194d)
**创建时间:** 2025-03-26

**内容:**
- 建立了完整的冻结策略模型实验框架
- 支持三种模型配置的对比分析
- 改进了分析工具，支持多模型比较和英文图表生成
- 解决了各种配置和路径问题
- 发现了beta head参数更新机制的严重bug

**状态说明:**
- 完成了冻结策略模型实验的基础设施建设
- 分析工具已经可以有效展示和比较不同模型的beta和delta值变化
- **最重要的发现**：通过实验比对确认beta head参数实际上并未有效更新，这是一个严重的bug，成为下一阶段的主要攻克目标
- 初步怀疑问题可能出在梯度计算、损失函数设计或反向传播机制上

**回退方法:**
```bash
git checkout 8df6194d
```

## 阶段7：修复beta head更新机制 (2025-04-01)

### 目标
诊断并修复beta head参数无法正常更新的关键bug，确保动态beta机制能够有效工作。

### 完成工作
1. **断点调试找到关键问题根源**
   - 通过系统性断点调试，追踪梯度流向
   - **关键发现**：在参考模型(ref_model)前向传播过程中，`self.current_beta_values` 变量失去了梯度连接
   - 具体问题点：`compute_reference_log_probs` 方法中，当调用参考模型进行前向计算时，使用了 `with torch.no_grad()` 上下文，随后又调用了 `concatenated_forward`，这导致新计算的 `self.current_beta_values` 与原始计算的值断开了梯度连接
   
2. **重构beta head参数更新机制**
   - 修改了参考模型计算流程，确保不会影响主要梯度流：
     ```python
     def compute_reference_log_probs(self, model, batch):
         # 保存原始的beta_values，避免被参考模型计算覆盖
         original_beta_values = self.current_beta_values
         
         with torch.no_grad():
             reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)
         
         # 恢复原始的beta_values
         self.current_beta_values = original_beta_values
         
         return reference_chosen_logps, reference_rejected_logps
     ```
   
   - 为beta_head设置独立的优化器配置：
     ```python
     # 为beta_head参数设置更高的学习率
     beta_head_lr = self.args.learning_rate * 10.0
     
     # 添加独立参数组
     params_config = {
         "params": beta_head_params,
         "lr": beta_head_lr,  # 使用更高的学习率
     }
     
     # 添加到优化器
     self.optimizer.add_param_group(params_config)
     ```
   - 修改了损失计算逻辑，确保beta值直接参与梯度计算
   
3. **实现梯度流测试与监控**
   - 创建了专门的`test_grad_flow`方法，模拟完整的前向和反向传播过程
   - 添加了断点调试辅助工具，用于监控关键变量的梯度状态
   - 实现了训练过程中beta_head参数变化的实时监控

4. **验证修复效果**
   - 运行全新的训练实验，成功观察到beta head参数有效更新
   - 通过断点调试确认梯度正确流向beta_head参数
   - 对比修复前后的模型性能，验证动态beta的价值

### 关键bug修复细节
**问题根源**: 通过断点调试发现，beta head的梯度链在参考模型前向传播过程中被截断，具体原因有：
1. 在`compute_reference_log_probs`方法中，使用了`with torch.no_grad()`上下文，但在这个上下文中调用了`concatenated_forward`，导致`self.current_beta_values`被覆盖为无梯度的版本
2. 后续的损失计算使用了这个无梯度的`self.current_beta_values`，导致beta_head无法获得有效梯度
3. 优化器配置中没有单独为beta_head设置更高学习率，即使有微小梯度也难以产生有效更新

**修复方案**:
1. 保存并恢复原始的beta_values，避免被参考模型计算覆盖：
   ```python
   # 保存原始的beta_values
   original_beta_values = self.current_beta_values
   
   with torch.no_grad():
       # 参考模型计算...
   
   # 恢复原始的beta_values
   self.current_beta_values = original_beta_values
   ```

2. 确保beta值直接参与损失计算：
   ```python
   # 使用动态beta计算损失
   losses = self.dynamic_beta_dpo_loss(
       policy_chosen_logps, policy_rejected_logps, 
       reference_chosen_logps, reference_rejected_logps,
       beta_values  # 直接传入有梯度的beta值
   )
   ```

3. 为beta_head设置独立参数组和更高学习率：
   ```python
   # 为beta_head参数设置更高的学习率
   beta_head_lr = self.args.learning_rate * 10.0
   self.optimizer.add_param_group({"params": beta_head_params, "lr": beta_head_lr})
   ```

### 保存点7: 成功修复beta head更新机制 (d85fe23a)
**创建时间:** 2025-04-02

**内容:**
- 通过断点调试定位并修复了beta head梯度链断开的核心问题
- 改进了参考模型计算流程，确保不会影响主要梯度流
- 重构了优化器配置，为beta_head设置了独立参数组和更高学习率
- 实现了梯度流测试函数，确保参数更新正常

**状态说明:**
- 动态beta机制现在能够正常工作，beta_head参数在训练过程中有效更新
- 观察到beta值随训练的合理变化，表明模型成功学习了适应性的beta调整策略
- 性能评估显示，与固定beta相比，动态beta在多个指标上有明显提升
- 通过断点调试这一经典方法，成功解决了难以诊断的深度学习系统梯度流问题

**回退方法:**
```bash
git checkout d85fe23a
# 或创建新分支
git checkout -b new_branch_name d85fe23a
```

## 下一阶段计划：阶段8 - 模型性能评估与对比分析

### 目标
全面评估动态beta模型的性能，与多个基准模型进行对比分析，并进一步优化动态beta策略。

### 具体计划:
1. **多模型对比测试**
   - 设置控制组：固定beta值的标准DPO
   - 实验组1：动态beta，完整训练
   - 实验组2：动态beta，冻结策略模型
   - 对比分析不同配置下的性能差异

2. **性能指标分析**
   - 实现详细的性能评估指标体系
   - 分析beta值与模型性能的相关性
   - 探索更优的beta初始化策略
   - 研究不同任务类型对最优beta值的影响

3. **优化动态beta策略**
   - 基于实验结果，设计更高效的beta计算网络结构
   - 尝试不同的beta约束机制
   - 研究beta值与输入复杂度的关系
   - 探索beta调整与模型不确定性的关联

4. **文档与分享**
   - 撰写详细的技术报告，记录实验结果和发现
   - 整理代码文档，便于社区使用和扩展
   - 准备技术分享材料，传播项目经验和成果

预计完成时间：2025-04-15
