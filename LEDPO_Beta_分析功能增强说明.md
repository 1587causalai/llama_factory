# LEDPO Beta 分析功能增强说明

## 功能简介

针对LEDPO（LearnableBetaDPO）训练过程中beta值的变化趋势监控和分析，我们增强了以下功能：

1. **实时监控**：在训练过程中实时记录beta相关指标
2. **可视化图表**：生成直观的beta趋势图表
3. **结果分析**：自动分析beta值是否存在问题并提供建议
4. **离线工具**：单独的分析脚本可用于分析已完成的训练结果

## 改进内容

### 1. 增强的LEDPOTrainer类

在`src/llamafactory/train/ledpo/trainer.py`中，我们对`LEDPOTrainer`类进行了以下增强：

- **beta_history记录**：添加了对beta_scale、pos_beta、neg_beta等指标的历史记录
- **图表绘制功能**：新增`plot_beta_trends`方法自动生成四面板趋势图
- **训练过程分析**：每隔一定步数打印beta分析信息
- **结果总结分析**：训练结束后自动生成beta分析总结

改进后的代码将：
- 在训练过程中实时记录beta指标
- 每100步自动生成一次图表
- 每50步打印一次详细分析
- 训练结束后生成总结报告和最终图表

### 2. 独立分析工具

#### analyze_ledpo_beta.py

在`scripts/analyze_ledpo_beta.py`中，我们创建了一个独立的分析工具：

- **加载数据**：可以读取训练过程中保存的beta历史数据
- **可视化分析**：生成多面板的综合分析图表
- **统计分析**：对beta相关指标进行统计分析
- **健康评估**：自动判断beta值是否存在问题并生成报告

#### run_ledpo_analysis.sh

在`scripts/run_ledpo_analysis.sh`中，我们提供了一个简单的脚本来运行分析工具：

- **自动检测**：检查所需的文件和目录
- **结果汇总**：展示关键的分析结果
- **可定制输出**：支持自定义输出目录和结果前缀

## 使用方法

### 1. 使用增强版Trainer训练

正常使用LLaMA-Factory进行LEDPO训练，训练过程中会自动记录数据并生成图表。
训练结束后会自动生成分析报告。

```bash
# 默认训练命令不变
python src/train_bash.py --train_type preference ...
```

### 2. 训练后分析

如果想对已完成的训练结果进行更详细的分析，可以使用独立的分析工具：

```bash
# 使用分析脚本
./scripts/run_ledpo_analysis.sh path/to/model_output_dir [可选前缀]
```

例如：

```bash
./scripts/run_ledpo_analysis.sh output/ledpo-qwen-7b-chat ledpo_qwen
```

## 分析指标说明

在分析过程中，我们主要关注以下指标：

- **beta_scale**：ValueHead中的beta缩放参数，应保持稳定
- **pos_beta**：正delta（即π(y)>π(y')）样本对应的beta值
- **neg_beta**：负delta（即π(y)<π(y')）样本对应的beta值
- **pos_neg_ratio**：pos_beta与neg_beta的比值，理想情况下应>1

## beta健康评估标准

我们使用以下标准来评估beta是否健康：

1. **无beta趋零问题**：beta值不应接近于beta_min
2. **良好的beta区分度**：pos_beta/neg_beta应大于1.2
3. **beta_scale稳定**：beta_scale不应有明显下降趋势

如果不满足以上标准，分析结果会给出相应的改进建议。

## 图表说明

生成的分析图表包含多个面板：

1. **Beta Scale趋势图**：显示beta_scale随训练步数的变化
2. **Pos/Neg Beta趋势图**：显示正负delta对应的beta值变化
3. **Beta比值趋势图**：显示pos_beta/neg_beta比值的变化
4. **损失变化图**：显示训练/验证损失的变化
5. **Beta分布热图**：显示pos_beta和neg_beta的分布关系
6. **Beta Scale关系图**：显示beta_scale与其他指标的关系
7. **统计摘要**：显示关键指标的统计结果和健康评估

## 常见问题解决方案

如果遇到beta趋零问题，可尝试以下解决方案：

1. 将`freeze_policy_model`设置为`False`
2. 增加`beta_min`参数值
3. 调整ValueHead的参数初始化
4. 对beta_scale添加正则化
5. 适当增加pos_beta与neg_beta的对比度

## 结论

通过这些增强功能，我们可以更好地监控和分析LEDPO训练过程中beta值的变化，及时发现并解决潜在问题，提高训练效果。 