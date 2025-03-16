# LeDPO 开发日志

本文档记录 LeDPO (可学习Beta DPO) 算法的渐进式开发过程，包括每个阶段的工作内容、决策和结果。

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

4. **存档当前进展**
   - 使用 Git 提交所有更改，创建第一个开发存档点
   - 提交信息："阶段1: 创建基本FooDPO实现作为渐进式开发的第一步"
   - 提交 ID: 4cd13c1800bfe2b82a98faf206a5fdfad11af655

### 下一步计划
准备进入阶段2：最小化LEDPO实现。将在现有的FooDPO基础上添加：
1. 简单的ValueHead网络来学习beta值
2. 修改loss计算，引入动态beta
3. 添加必要的监控代码来跟踪beta值的变化

### 运行命令
```bash
llamafactory-cli train ledpo_progressive_dev/qwen15_lora_foodpo.yaml
```

### 注意事项
- 当前实现仍然是标准DPO，只是重命名为FooDPO，作为后续开发的基础
- 确保所有的变更都遵循渐进式开发的原则：可控、可测试、简洁、有记录、可回退 

## 训练指标监控实现

为了更好地监控LEDPO算法的训练过程，我们创建了一个专门的绘图脚本，用于可视化训练过程中的关键指标。

### 关键指标

根据项目需求，我们特别关注以下6个指标，并按此顺序显示：

1. **accuracy** (rewards/accuracies)
2. **loss**
3. **reward/margin** (rewards/margins)
4. **reward/chosen** (rewards/chosen)
5. **reward/rejected** (rewards/rejected)
6. **beta** (beta参数)

### 新增LEDPO特定指标支持

为了准备LEDPO算法的实现，我们扩展了绘图脚本，增加了对以下新指标的支持：

1. **pos_beta** - 正样本的beta值
2. **neg_beta** - 负样本的beta值

这些指标对于LEDPO的开发至关重要，因为它们直接反映了算法学习beta值的效果：

- pos_beta值应该较高，表明模型对正样本的高置信度
- neg_beta值应该较低，表明模型对负样本的低置信度
- 两者之间的差距越大，表明模型区分能力越强

通过在同一图表中显示这两个值的变化趋势，我们可以直观地监控LEDPO算法的学习效果。这些改动采用了最小化原则，只在必要的地方添加了新指标的支持，保持了原有功能的完整性。

### 实现方案

我们采用了最小改动原则，创建了两个独立的Python脚本：

1. **plot_ledpo_metrics.py**：
   - 直接从`trainer_state.json`中读取训练和评估数据
   - 在同一张图上绘制train和eval数据，使用不同的线条区分
   - 按照指定的顺序排列6个关键指标
   - 生成高质量的可视化图表，便于分析模型训练效果

2. **run_train_and_plot.py**：
   - 集成训练和绘图为一体的工作流
   - 先运行标准训练过程
   - 训练完成后自动调用绘图脚本生成指标图表
   - 支持命令行参数自定义配置

这种方案的优点：
- 不修改LlamaFactory的任何源代码
- 不使用侵入式的回调机制
- 完全独立的后处理方案，更加稳定可靠
- 可以随时应用于已完成的训练结果
- 便于在不同实验间进行指标对比

### 使用方法

训练完成后，可以使用以下命令生成指标图表：

```bash
python ledpo_progressive_dev/plot_ledpo_metrics.py --result_dir results/qwen15-0.5b/lora/foodpo
```

或者使用集成脚本一键完成训练和绘图：

```bash
python ledpo_progressive_dev/run_train_and_plot.py --config ledpo_progressive_dev/qwen15_lora_foodpo.yaml
```

生成的图表保存在训练输出目录的`ledpo_plots`子目录中。 