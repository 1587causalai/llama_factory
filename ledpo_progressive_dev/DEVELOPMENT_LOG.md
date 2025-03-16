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