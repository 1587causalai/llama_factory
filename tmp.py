import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# 定义 logits 的范围
logits = np.linspace(-5, 5, 100)  # 从 -5 到 5，取 100 个点

# 选择不同的 beta 值
betas = [0.1, 0.5, 1.0, 2.0]  # 4 个不同的 beta 值


plt.figure(figsize=(10, 8))
# 创建 2x2 子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 创建一个 2x2 的子图网格
axes = axes.flatten()  # 将 axes 展平成一维数组，方便遍历

# 遍历 beta 值并绘制子图
for i, beta in enumerate(betas):
    # 计算原始 DPO 的偏好概率（sigmoid）
    p_original = 1 / (1 + np.exp(-beta * logits))
    
    # 计算改进后的偏好概率（erf）
    p_new = 0.5 * (1 + erf(0.6 * beta * logits / np.sqrt(2)))  # 0.6 是经验值
    
    # 绘制曲线
    axes[i].plot(logits, p_original, label='Original DPO (sigmoid)', color='blue')
    axes[i].plot(logits, p_new, label='New DPO (erf)', color='red', linestyle='--')
    axes[i].set_title(f'beta = {beta}')  # 设置子图标题
    axes[i].set_xlabel('logits')  # x 轴标签
    axes[i].set_ylabel('Preference Probability')  # y 轴标签
    axes[i].legend()  # 添加图例
    axes[i].grid(True)  # 添加网格线

# 调整子图间距
plt.tight_layout()
# plt.show()
plt.savefig('xdpo/imgs/disco_vs_sigmoid_pref_prob.png')