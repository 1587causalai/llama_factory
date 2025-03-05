# fooDPO算法实现

这个项目提供了fooDPO (Food Direct Preference Optimization) 算法的实现，这是一种基于DPO的改进算法，用于从人类偏好数据中优化语言模型。项目将fooDPO训练过程拆分为前向传播、动态权重计算、损失计算和反向传播四个阶段，便于深入理解fooDPO工作原理。

fooDPO算法在标准DPO的基础上引入了以下创新点：
1. 动态beta值调整 - 通过`pref_beta_scale`参数动态调整不同样本的beta值
2. 特殊的`foo_factor`因子 - 为算法引入新的权重调整机制
3. 动态权重计算 - 通过`dynamic_weight`开关启用样本级别的动态权重

## 运行方法

执行以下命令启动fooDPO训练：

```bash
llamafactory-cli train examples/train_lora/qwen1_5_0_5b_lora_fooDPO.yaml
```

或者使用提供的脚本：

```bash
./run_foodpo.sh
```


