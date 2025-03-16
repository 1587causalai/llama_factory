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
| 6    | beta值与动态优化实验 | 计划中 | - |

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
git checkout ec6b82c1
```

## 下一阶段计划：阶段6 (计划中)

### 目标
在阶段5的基础上，开展更全面的实验，比较冻结与非冻结模型的表现差异，探索不同学习率和训练轮次的效果。

### 具体计划:
1. 对比冻结和非冻结模型的表现差异
2. 实验不同的beta_head学习率
3. 探索更长训练轮次下beta值的变化趋势
4. 分析delta和beta值之间的关系
