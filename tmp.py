import json
import matplotlib.pyplot as plt
import numpy as np

# 加载trainer_state.json
with open('saves/qwen/ledpo_dynamic_beta_freeze_policy_model/trainer_state.json', 'r') as f:
    data = json.load(f)

# 提取训练步骤和对应的beta值
steps = []
pos_betas = []
neg_betas = []
beta_scales = []
deltas = []

for log in data['log_history']:
    # 只提取训练步骤的数据（不包括eval）
    if 'beta/positive_delta_avg' in log and 'eval_' not in str(log):
        steps.append(log.get('step', 0))
        pos_betas.append(log.get('beta/positive_delta_avg', 0))
        neg_betas.append(log.get('beta/negative_delta_avg', 0))
        beta_scales.append(log.get('beta/scale', 0))
        deltas.append(log.get('delta/mean', 0))

# 创建绘图
plt.figure(figsize=(12, 8))

# 绘制beta值变化
plt.subplot(2, 1, 1)
plt.plot(steps, pos_betas, 'g-', label='Positive Delta Beta')
plt.plot(steps, neg_betas, 'r-', label='Negative Delta Beta')
plt.plot(steps, beta_scales, 'b--', label='Beta Scale')
plt.xlabel('Training Steps')
plt.ylabel('Beta Value')
plt.title('Beta Values During Training')
plt.legend()
plt.grid(True)

# 绘制delta平均值
plt.subplot(2, 1, 2)
plt.plot(steps, deltas, 'k-', label='Mean Delta')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.xlabel('Training Steps')
plt.ylabel('Delta Value')
plt.title('Mean Delta During Training')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('beta_analysis.png')
plt.show()

# 打印统计信息
print(f"初始正delta beta: {pos_betas[0]:.6f}, 最终: {pos_betas[-1]:.6f}, 变化: {pos_betas[-1] - pos_betas[0]:.6f}")
print(f"初始负delta beta: {neg_betas[0]:.6f}, 最终: {neg_betas[-1]:.6f}, 变化: {neg_betas[-1] - neg_betas[0]:.6f}")
print(f"初始beta scale: {beta_scales[0]:.6f}, 最终: {beta_scales[-1]:.6f}, 变化: {beta_scales[-1] - beta_scales[0]:.6f}")

# 计算相关系数，检查delta与beta的关系
if len(deltas) > 1:
    pos_corr = np.corrcoef(deltas, pos_betas)[0,1]
    neg_corr = np.corrcoef(deltas, neg_betas)[0,1]
    print(f"Delta与正delta beta相关系数: {pos_corr:.4f}")
    print(f"Delta与负delta beta相关系数: {neg_corr:.4f}")

# 计算delta>0和delta<0情况下的平均beta值
pos_delta_indices = [i for i, d in enumerate(deltas) if d > 0]
neg_delta_indices = [i for i, d in enumerate(deltas) if d <= 0]

if pos_delta_indices:
    pos_delta_pos_beta = [pos_betas[i] for i in pos_delta_indices]
    pos_delta_neg_beta = [neg_betas[i] for i in pos_delta_indices]
    print(f"Delta>0时，正delta beta平均值: {np.mean(pos_delta_pos_beta):.6f}")
    print(f"Delta>0时，负delta beta平均值: {np.mean(pos_delta_neg_beta):.6f}")

if neg_delta_indices:
    neg_delta_pos_beta = [pos_betas[i] for i in neg_delta_indices]
    neg_delta_neg_beta = [neg_betas[i] for i in neg_delta_indices]
    print(f"Delta<0时，正delta beta平均值: {np.mean(neg_delta_pos_beta):.6f}")
    print(f"Delta<0时，负delta beta平均值: {np.mean(neg_delta_neg_beta):.6f}")