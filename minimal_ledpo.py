#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
极简版LearnableBetaDPO实现

这个脚本提供了一个最简化的LearnableBetaDPO实现，
去除了所有复杂性，只保留核心逻辑，以便调试beta趋零问题。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Tuple, Union, Any

# 创建目录保存结果
os.makedirs("results", exist_ok=True)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

#######################
# 1. 定义ValueHead模块 #
#######################

class MinimalValueHead(nn.Module):
    """
    最简化的ValueHead实现
    """
    def __init__(self, hidden_size: int = 32, beta_min: float = 0.01, beta_max: float = 100.0):
        super().__init__()
        # beta_scale是全局缩放因子
        self.beta_scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        
        # 简单的前馈网络
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()  # 确保输出为正值
        )
        
        # beta的范围约束
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # 初始化网络权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.constant_(module.bias, 0.1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 每个样本预测原始beta值
        raw_beta = self.value_head(hidden_states)
        
        # 应用全局缩放
        scaled_beta = self.beta_scale * raw_beta
        
        # 限制beta范围
        clamped_beta = torch.clamp(scaled_beta, min=self.beta_min, max=self.beta_max)
        
        return clamped_beta


######################
# 2. 定义模拟数据生成 #
######################

def generate_synthetic_data(n_samples: int = 100, hidden_dim: int = 32):
    """
    生成合成数据用于训练和测试LEDPO
    
    返回:
        chosen_hidden: 形状为[n_samples, hidden_dim]的隐藏状态
        chosen_logps: 形状为[n_samples]的log概率
        rejected_logps: 形状为[n_samples]的log概率
        ref_chosen_logps: 形状为[n_samples]的参考log概率
        ref_rejected_logps: 形状为[n_samples]的参考log概率
        delta: 形状为[n_samples]的delta值
    """
    # 生成隐藏状态
    chosen_hidden = torch.randn(n_samples, hidden_dim)
    
    # 生成策略模型的log概率
    # 这里我们使控制chosen和rejected的差值，使得delta有正有负
    chosen_logps = torch.randn(n_samples) * 0.5
    rejected_logps = chosen_logps - torch.randn(n_samples) * 0.3
    
    # 生成参考模型的log概率
    ref_chosen_logps = chosen_logps - torch.randn(n_samples) * 0.2
    ref_rejected_logps = rejected_logps - torch.randn(n_samples) * 0.2
    
    # 计算delta
    chosen_logratios = chosen_logps - ref_chosen_logps
    rejected_logratios = rejected_logps - ref_rejected_logps
    delta = chosen_logratios - rejected_logratios
    
    # 打印数据分布信息
    print(f"Delta分布: 最小值={delta.min().item():.4f}, 最大值={delta.max().item():.4f}, 平均值={delta.mean().item():.4f}")
    print(f"Delta>0的样本数: {(delta > 0).sum().item()}, Delta<0的样本数: {(delta < 0).sum().item()}")
    
    return {
        "chosen_hidden": chosen_hidden, 
        "chosen_logps": chosen_logps,
        "rejected_logps": rejected_logps,
        "ref_chosen_logps": ref_chosen_logps,
        "ref_rejected_logps": ref_rejected_logps,
        "delta": delta
    }


########################
# 3. 定义LEDPO损失计算 #
########################

def compute_ledpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算LEDPO损失
    
    Args:
        policy_chosen_logps: 策略模型对chosen回复的log概率
        policy_rejected_logps: 策略模型对rejected回复的log概率
        reference_chosen_logps: 参考模型对chosen回复的log概率
        reference_rejected_logps: 参考模型对rejected回复的log概率
        beta: beta值，可以是标量或张量

    Returns:
        loss: DPO损失
        chosen_rewards: chosen奖励
        rejected_rewards: rejected奖励
    """
    # 计算logits (log-odds)
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps
    logits = beta * (chosen_logratios - rejected_logratios)
    
    # 计算损失 - 基本的DPO损失
    losses = -F.logsigmoid(logits)
    
    # 计算奖励
    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios
    
    return losses, chosen_rewards, rejected_rewards


######################
# 4. 训练和评估函数 #
######################

def train_minimal_ledpo(
    value_head: MinimalValueHead,
    train_data: Dict[str, torch.Tensor],
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    freeze_policy_model: bool = False
) -> Dict[str, List[float]]:
    """
    训练最简化版本的LEDPO
    
    Args:
        value_head: ValueHead模型
        train_data: 训练数据字典
        num_epochs: 训练轮数
        learning_rate: 学习率
        freeze_policy_model: 是否冻结策略模型(这里只影响梯度计算)
        
    Returns:
        metrics: 包含训练过程指标的字典
    """
    # 解包训练数据
    chosen_hidden = train_data["chosen_hidden"]
    chosen_logps = train_data["chosen_logps"]
    rejected_logps = train_data["rejected_logps"]
    ref_chosen_logps = train_data["ref_chosen_logps"]
    ref_rejected_logps = train_data["ref_rejected_logps"]
    delta = train_data["delta"]
    
    # 创建优化器
    optimizer = optim.Adam(value_head.parameters(), lr=learning_rate)
    
    # 用于记录指标
    metrics = {
        "loss": [],
        "beta_scale": [],
        "beta_mean": [],
        "positive_beta_avg": [],
        "negative_beta_avg": []
    }
    
    for epoch in range(num_epochs):
        # 清除梯度
        optimizer.zero_grad()
        
        # 1. 计算beta值
        beta = value_head(chosen_hidden).squeeze(-1)
        
        # 2. 计算delta值
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        curr_delta = chosen_logratios - rejected_logratios
        
        # 3. 分离正负delta样本
        positive_delta_mask = (curr_delta > 0)
        negative_delta_mask = (curr_delta < 0)
        
        # 4. 计算损失
        if freeze_policy_model:
            # 如果冻结策略模型，chosen_logps和rejected_logps不参与梯度计算
            # 使用detach()而不是torch.no_grad()，这样可以保留beta的梯度
            chosen_logps_detached = chosen_logps.detach()
            rejected_logps_detached = rejected_logps.detach()
            losses, _, _ = compute_ledpo_loss(
                chosen_logps_detached, 
                rejected_logps_detached,
                ref_chosen_logps,
                ref_rejected_logps,
                beta
            )
        else:
            # 正常计算损失
            losses, _, _ = compute_ledpo_loss(
                chosen_logps, 
                rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta
            )
        
        # 5. 计算平均损失
        loss = losses.mean()
        
        # 6. 反向传播计算梯度
        loss.backward()
        
        # 7. 更新参数
        optimizer.step()
        
        # 8. 记录指标
        metrics["loss"].append(loss.item())
        metrics["beta_scale"].append(value_head.beta_scale.item())
        metrics["beta_mean"].append(beta.mean().item())
        
        # 计算正负delta样本对应的beta平均值
        pos_count = positive_delta_mask.sum().item()
        neg_count = negative_delta_mask.sum().item()
        
        if pos_count > 0:
            positive_beta_avg = beta[positive_delta_mask].mean().item()
        else:
            positive_beta_avg = 0.0
            
        if neg_count > 0:
            negative_beta_avg = beta[negative_delta_mask].mean().item()
        else:
            negative_beta_avg = 0.0
            
        metrics["positive_beta_avg"].append(positive_beta_avg)
        metrics["negative_beta_avg"].append(negative_beta_avg)
        
        # 9. 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: loss={loss.item():.4f}, beta_scale={value_head.beta_scale.item():.4f}, "
                  f"beta_mean={beta.mean().item():.4f}, pos_beta={positive_beta_avg:.4f}, neg_beta={negative_beta_avg:.4f}")
    
    return metrics


######################
# 5. 结果可视化函数 #
######################

def plot_results(metrics: Dict[str, List[float]], freeze_policy_model: bool = False):
    """
    绘制训练结果
    
    Args:
        metrics: 包含训练过程指标的字典
        freeze_policy_model: 是否冻结了策略模型
    """
    epochs = range(1, len(metrics["loss"]) + 1)
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 1. 绘制损失
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics["loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    
    # 2. 绘制beta_scale
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics["beta_scale"])
    plt.xlabel("Epochs")
    plt.ylabel("Beta Scale")
    plt.title("Beta Scale")
    plt.grid(True)
    
    # 3. 绘制平均beta值
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics["beta_mean"])
    plt.xlabel("Epochs")
    plt.ylabel("Mean Beta")
    plt.title("Average Beta Value")
    plt.grid(True)
    
    # 4. 绘制正负delta对应的beta值
    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics["positive_beta_avg"], label="Beta for Delta > 0")
    plt.plot(epochs, metrics["negative_beta_avg"], label="Beta for Delta < 0")
    plt.xlabel("Epochs")
    plt.ylabel("Beta Value")
    plt.title("Beta Values for Positive/Negative Delta")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图形
    title = "freeze_policy_model" if freeze_policy_model else "normal"
    plt.savefig(f"results/minimal_ledpo_{title}.png")
    plt.close()


######################
# 6. 主函数 #
######################

def main():
    """
    运行最小化版本的LEDPO训练
    """
    # 设置参数
    hidden_dim = 32
    n_samples = 100
    num_epochs = 200
    learning_rate = 1e-3
    
    # 生成训练数据
    train_data = generate_synthetic_data(n_samples=n_samples, hidden_dim=hidden_dim)
    
    print("\n===== 测试场景1: 不冻结策略模型 =====")
    # 创建ValueHead模型
    value_head1 = MinimalValueHead(hidden_size=hidden_dim)
    
    # 训练模型
    metrics1 = train_minimal_ledpo(
        value_head=value_head1,
        train_data=train_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        freeze_policy_model=False
    )
    
    # 绘制结果
    plot_results(metrics1, freeze_policy_model=False)
    
    print("\n===== 测试场景2: 冻结策略模型 =====")
    # 创建ValueHead模型
    value_head2 = MinimalValueHead(hidden_size=hidden_dim)
    
    # 训练模型
    metrics2 = train_minimal_ledpo(
        value_head=value_head2,
        train_data=train_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        freeze_policy_model=True
    )
    
    # 绘制结果
    plot_results(metrics2, freeze_policy_model=True)
    
    print("\n===== 分析结果 =====")
    print("不冻结策略模型:")
    print(f"  最终beta_scale = {metrics1['beta_scale'][-1]:.4f}")
    print(f"  最终pos_beta/neg_beta比值 = {metrics1['positive_beta_avg'][-1]/max(1e-5, metrics1['negative_beta_avg'][-1]):.4f}")
    
    print("冻结策略模型:")
    print(f"  最终beta_scale = {metrics2['beta_scale'][-1]:.4f}")
    print(f"  最终pos_beta/neg_beta比值 = {metrics2['positive_beta_avg'][-1]/max(1e-5, metrics2['negative_beta_avg'][-1]):.4f}")
    
    # 将结果保存为文本文件
    with open("results/minimal_ledpo_results.txt", "w") as f:
        f.write("===== 不冻结策略模型 =====\n")
        f.write(f"最终beta_scale = {metrics1['beta_scale'][-1]:.4f}\n")
        f.write(f"最终pos_beta = {metrics1['positive_beta_avg'][-1]:.4f}\n")
        f.write(f"最终neg_beta = {metrics1['negative_beta_avg'][-1]:.4f}\n")
        f.write(f"最终pos_beta/neg_beta比值 = {metrics1['positive_beta_avg'][-1]/max(1e-5, metrics1['negative_beta_avg'][-1]):.4f}\n\n")
        
        f.write("===== 冻结策略模型 =====\n")
        f.write(f"最终beta_scale = {metrics2['beta_scale'][-1]:.4f}\n")
        f.write(f"最终pos_beta = {metrics2['positive_beta_avg'][-1]:.4f}\n")
        f.write(f"最终neg_beta = {metrics2['negative_beta_avg'][-1]:.4f}\n")
        f.write(f"最终pos_beta/neg_beta比值 = {metrics2['positive_beta_avg'][-1]/max(1e-5, metrics2['negative_beta_avg'][-1]):.4f}\n")


if __name__ == "__main__":
    main() 