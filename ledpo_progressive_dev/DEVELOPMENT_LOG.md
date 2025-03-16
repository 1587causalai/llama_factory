# LeDPO 开发日志

本文档记录 LeDPO (可学习Beta DPO) 算法的渐进式开发过程，包括每个阶段的工作内容、决策和结果。

## 开发阶段概览

| 阶段 | 描述 | 状态 | 对应保存点 |
|------|------|------|------------|
| 1    | 标准DPO基准测试 | 已完成 | 保存点1 |
| 2    | LEDPO基础框架建立 | 已完成 | 保存点2 |
| 3    | 基于最后提示词Token的动态beta实现 | 已完成 | 保存点3 |
| 4    | 改进动态beta值监控系统 | 已完成 | 保存点4 |
| 5    | 将动态beta应用到损失计算中 | 计划中 | - |

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

## 阶段2：LEDPO基础框架建立 (2025-03-18)

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
**创建时间:** 2025-03-19

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
**创建时间:** 2025-03-20

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

## 下一阶段计划：阶段5 (计划中)

在前四个阶段的基础上，我们已经完成了动态beta计算和监控系统的建设。下一步将进入核心阶段：将动态beta值真正应用到损失计算中。

### 具体计划:
1. 修改`compute_preference_loss`函数，将动态beta值应用到实际损失计算中
2. 实现不同损失函数(如DPO、ORPO、SimPO)下的动态beta应用方案
3. 对比实验验证动态beta的效果
4. 添加beta值范围限制和正则化策略，防止beta趋零或爆炸
5. 实现beta值的可视化和分析工具 