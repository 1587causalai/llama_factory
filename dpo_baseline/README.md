# DPO基线训练实现

这个项目提供了一个完整的Direct Preference Optimization (DPO) 训练基线实现，用于从人类偏好数据中优化语言模型。项目将DPO训练过程拆分为前向传播、损失计算和反向传播三个阶段，便于深入理解DPO工作原理，为开发自定义的偏好优化算法（如FooDPO）做准备。



我正在进行一项研究，


```bash
llamafactory-cli train examples/train_lora/qwen1_5_0_5b_lora_dpo_test.yaml
``` 


