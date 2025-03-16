#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
极简版LearnableBetaDPO变体实现

这个脚本提供了几种不同变体的LearnableBetaDPO最简实现，
用于测试不同解决beta趋零问题的方案。
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


#########################
# 1. 定义ValueHead变体 #
#########################

class MinimalValueHead(nn.Module):
    """
    基础版ValueHead实现
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


class SoftplusValueHead(nn.Module):
    """
    使用Softplus激活beta_scale的ValueHead实现
    """
    def __init__(self, hidden_size: int = 32, beta_min: float = 0.01, beta_max: float = 100.0):
        super().__init__()
        # beta_scale是全局缩放因子，使用更大的初始值
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
        """前向传播 - 使用softplus确保beta_scale始终为正"""
        # 每个样本预测原始beta值
        raw_beta = self.value_head(hidden_states)
        
        # 应用softplus确保beta_scale始终为正值
        beta_scale_positive = F.softplus(self.beta_scale)
        
        # 应用全局缩放
        scaled_beta = beta_scale_positive * raw_beta
        
        # 限制beta范围
        clamped_beta = torch.clamp(scaled_beta, min=self.beta_min, max=self.beta_max)
        
        return clamped_beta


class DeltaAwareValueHead(nn.Module):
    """
    Delta感知型ValueHead，根据delta符号直接调整beta
    """
    def __init__(self, hidden_size: int = 32, beta_min: float = 0.01, beta_max: float = 100.0):
        super().__init__()
        # beta基础值参数
        self.beta_base = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
        # 简单的前馈网络，输出一个系数
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出0-1之间的值
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
    
    def forward(self, hidden_states: torch.Tensor, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - 根据delta符号调整beta
        
        Args:
            hidden_states: 隐藏状态
            delta: 可选的delta值，用于调整beta
        """
        # 每个样本预测原始beta系数
        beta_coef = self.value_head(hidden_states)
        
        # 如果提供了delta，根据delta符号调整beta
        if delta is not None:
            # 将delta符号转换为权重 (delta>0时为1.2, delta<0时为0.8)
            delta_weight = torch.where(delta > 0, 
                                      torch.tensor(1.2, device=delta.device), 
                                      torch.tensor(0.8, device=delta.device))
            # 应用权重
            beta = self.beta_base * beta_coef * delta_weight
        else:
            beta = self.beta_base * beta_coef
        
        # 限制beta范围
        clamped_beta = torch.clamp(beta, min=self.beta_min, max=self.beta_max)
        
        return clamped_beta


######################
# 2. 定义模拟数据生成 #
######################

def generate_synthetic_data(n_samples: int = 100, hidden_dim: int = 32, delta_ratio: float = 0.6):
    """
    生成合成数据用于训练和测试LEDPO
    
    Args:
        n_samples: 样本数量
        hidden_dim: 隐藏状态维度
        delta_ratio: delta>0的样本比例 (0-1之间)
        
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
    chosen_logps = torch.randn(n_samples) * 0.5
    rejected_logps = torch.randn(n_samples) * 0.5
    
    # 生成参考模型的log概率
    ref_chosen_logps = torch.randn(n_samples) * 0.5
    ref_rejected_logps = torch.randn(n_samples) * 0.5
    
    # 计算初始delta
    chosen_logratios = chosen_logps - ref_chosen_logps
    rejected_logratios = rejected_logps - ref_rejected_logps
    delta = chosen_logratios - rejected_logratios
    
    # 调整delta，使得delta_ratio比例的样本delta>0
    # 首先对delta进行排序
    sorted_indices = torch.argsort(delta)
    n_positive = int(n_samples * delta_ratio)
    
    # 找到阈值
    threshold_idx = sorted_indices[n_samples - n_positive]
    threshold = delta[threshold_idx]
    
    # 根据阈值调整delta
    # 对于需要为正的样本，如果已经为正，不变；如果为负，加上一个偏移使其变为正
    # 对于需要为负的样本，如果已经为负，不变；如果为正，减去一个偏移使其变为负
    for i in range(n_samples):
        if i >= n_samples - n_positive and delta[i] < 0:  # 应该为正但实际为负
            # 调整参考模型的log概率，使delta变为正
            adjustment = abs(delta[i]) + 0.1  # 加上一点余量
            ref_rejected_logps[i] += adjustment
        elif i < n_samples - n_positive and delta[i] > 0:  # 应该为负但实际为正
            # 调整参考模型的log概率，使delta变为负
            adjustment = abs(delta[i]) + 0.1  # 加上一点余量
            ref_chosen_logps[i] += adjustment
    
    # 重新计算delta
    chosen_logratios = chosen_logps - ref_chosen_logps
    rejected_logratios = rejected_logps - ref_rejected_logps
    delta = chosen_logratios - rejected_logratios
    
    # 打印数据分布信息
    print(f"Delta分布: 最小值={delta.min().item():.4f}, 最大值={delta.max().item():.4f}, 平均值={delta.mean().item():.4f}")
    print(f"Delta>0的样本数: {(delta > 0).sum().item()}, Delta<0的样本数: {(delta < 0).sum().item()}")
    print(f"Delta>0的样本比例: {(delta > 0).sum().item() / n_samples:.2f}")
    
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
    beta: torch.Tensor,
    use_regularization: bool = False,
    beta_scale: Optional[torch.Tensor] = None,
    beta_reg_factor: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算LEDPO损失
    
    Args:
        policy_chosen_logps: 策略模型对chosen回复的log概率
        policy_rejected_logps: 策略模型对rejected回复的log概率
        reference_chosen_logps: 参考模型对chosen回复的log概率
        reference_rejected_logps: 参考模型对rejected回复的log概率
        beta: beta值，可以是标量或张量
        use_regularization: 是否使用beta_scale正则化
        beta_scale: beta_scale参数，仅在use_regularization=True时使用
        beta_reg_factor: beta正则化系数

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
    
    # 如果启用正则化，添加beta_scale的正则化项
    if use_regularization and beta_scale is not None:
        # 防止beta_scale变得过小的正则化
        beta_reg_loss = beta_reg_factor * F.relu(1.0 - beta_scale)
        # 将正则化损失添加到每个样本的损失上
        losses = losses + beta_reg_loss
    
    # 计算奖励
    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios
    
    return losses, chosen_rewards, rejected_rewards


######################
# 4. 训练和评估函数 #
######################

def train_ledpo_variant(
    value_head: nn.Module,
    train_data: Dict[str, torch.Tensor],
    variant_name: str,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    freeze_policy_model: bool = False,
    use_softplus: bool = False,
    use_regularization: bool = False,
    delta_aware: bool = False
) -> Dict[str, List[float]]:
    """
    训练各种变体的LEDPO
    
    Args:
        value_head: ValueHead模型
        train_data: 训练数据字典
        variant_name: 变体名称，用于日志记录
        num_epochs: 训练轮数
        learning_rate: 学习率
        freeze_policy_model: 是否冻结策略模型
        use_softplus: 是否使用softplus激活beta_scale
        use_regularization: 是否使用beta_scale正则化
        delta_aware: 是否使用delta感知型ValueHead
        
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
        if delta_aware and hasattr(value_head, "forward") and "delta" in value_head.forward.__code__.co_varnames:
            beta = value_head(chosen_hidden, delta).squeeze(-1)
        else:
            beta = value_head(chosen_hidden).squeeze(-1)
        
        # 2. 计算当前delta值
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        curr_delta = chosen_logratios - rejected_logratios
        
        # 3. 分离正负delta样本
        positive_delta_mask = (curr_delta > 0)
        negative_delta_mask = (curr_delta < 0)
        
        # 4. 获取beta_scale用于正则化和记录
        if hasattr(value_head, 'beta_scale'):
            if use_softplus:
                beta_scale_value = F.softplus(value_head.beta_scale)
            else:
                beta_scale_value = value_head.beta_scale
        elif hasattr(value_head, 'beta_base'):
            beta_scale_value = value_head.beta_base
        else:
            beta_scale_value = torch.tensor(1.0)
        
        # 5. 计算损失
        if freeze_policy_model:
            # 如果冻结策略模型，chosen_logps和rejected_logps不参与梯度计算
            # 使用detach()而不是torch.no_grad()，这样可以保留beta的梯度
            chosen_logps_detached = chosen_logps.detach()
            rejected_logps_detached = rejected_logps.detach()
            if use_regularization:
                losses, _, _ = compute_ledpo_loss(
                    chosen_logps_detached, 
                    rejected_logps_detached,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta,
                    use_regularization=True,
                    beta_scale=beta_scale_value
                )
            else:
                losses, _, _ = compute_ledpo_loss(
                    chosen_logps_detached, 
                    rejected_logps_detached,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta
                )
        else:
            # 正常计算损失
            if use_regularization:
                losses, _, _ = compute_ledpo_loss(
                    chosen_logps, 
                    rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta,
                    use_regularization=True,
                    beta_scale=beta_scale_value
                )
            else:
                losses, _, _ = compute_ledpo_loss(
                    chosen_logps, 
                    rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta
                )
        
        # 6. 计算平均损失
        loss = losses.mean()
        
        # 7. 反向传播计算梯度
        loss.backward()
        
        # 8. 更新参数
        optimizer.step()
        
        # 9. 记录指标
        metrics["loss"].append(loss.item())
        metrics["beta_scale"].append(beta_scale_value.item())
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
        
        # 10. 打印进度
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"{variant_name} - Epoch {epoch+1}/{num_epochs}: loss={loss.item():.4f}, "
                  f"beta_scale={beta_scale_value.item():.4f}, beta_mean={beta.mean().item():.4f}, "
                  f"pos_beta={positive_beta_avg:.4f}, neg_beta={negative_beta_avg:.4f}")
    
    return metrics


######################
# 5. 结果可视化函数 #
######################

def plot_results(metrics_dict: Dict[str, Dict[str, List[float]]], title: str = "LEDPO Variants"):
    """
    绘制多个训练变体的结果对比
    
    Args:
        metrics_dict: 包含不同变体指标的字典，键为变体名称
        title: 图表标题
    """
    variants = list(metrics_dict.keys())
    # 确保所有变体有相同数量的epoch
    epochs = range(1, len(metrics_dict[variants[0]]["loss"]) + 1)
    
    # 创建图形
    plt.figure(figsize=(15, 15))
    
    # 1. 绘制损失
    plt.subplot(2, 2, 1)
    for variant in variants:
        plt.plot(epochs, metrics_dict[variant]["loss"], label=variant)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    
    # 2. 绘制beta_scale
    plt.subplot(2, 2, 2)
    for variant in variants:
        plt.plot(epochs, metrics_dict[variant]["beta_scale"], label=variant)
    plt.xlabel("Epochs")
    plt.ylabel("Beta Scale")
    plt.title("Beta Scale Value")
    plt.legend()
    plt.grid(True)
    
    # 3. 绘制平均beta值
    plt.subplot(2, 2, 3)
    for variant in variants:
        plt.plot(epochs, metrics_dict[variant]["beta_mean"], label=variant)
    plt.xlabel("Epochs")
    plt.ylabel("Mean Beta")
    plt.title("Average Beta Value")
    plt.legend()
    plt.grid(True)
    
    # 4. 绘制正负delta的beta比值
    plt.subplot(2, 2, 4)
    for variant in variants:
        # 计算比值，避免除以0
        ratios = []
        for i in range(len(epochs)):
            neg_beta = max(1e-5, metrics_dict[variant]["negative_beta_avg"][i])  # 避免除以0
            pos_beta = metrics_dict[variant]["positive_beta_avg"][i]
            ratios.append(pos_beta / neg_beta)
        plt.plot(epochs, ratios, label=variant)
    plt.xlabel("Epochs")
    plt.ylabel("Pos Beta / Neg Beta Ratio")
    plt.title("Ratio of Positive to Negative Delta Beta")
    plt.legend()
    plt.grid(True)
    
    plt.suptitle("Comparison of Different LEDPO Variants", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间
    
    # 保存图形
    safe_title = title.replace(" ", "_").lower()
    plt.savefig(f"results/{safe_title}.png")
    plt.close()


######################
# 6. 主函数 #
######################

def main():
    """
    运行多个LEDPO变体的实验
    """
    # 设置参数
    hidden_dim = 32
    n_samples = 100
    num_epochs = 300
    learning_rate = 1e-3
    
    # 生成训练数据 - 确保delta的正负比例为60:40
    train_data = generate_synthetic_data(n_samples=n_samples, hidden_dim=hidden_dim, delta_ratio=0.6)
    
    # 定义要测试的变体配置
    variants = [
        {
            "name": "Basic-UnfrozenPolicy", 
            "head_class": MinimalValueHead,
            "freeze_policy": False,
            "use_softplus": False,
            "use_regularization": False,
            "delta_aware": False
        },
        {
            "name": "Basic-FrozenPolicy", 
            "head_class": MinimalValueHead,
            "freeze_policy": True,
            "use_softplus": False,
            "use_regularization": False,
            "delta_aware": False
        },
        {
            "name": "Softplus-FrozenPolicy",
            "head_class": SoftplusValueHead,
            "freeze_policy": True,
            "use_softplus": True,
            "use_regularization": False,
            "delta_aware": False
        },
        {
            "name": "Regularized-FrozenPolicy",
            "head_class": MinimalValueHead,
            "freeze_policy": True,
            "use_softplus": False,
            "use_regularization": True,
            "delta_aware": False
        },
        {
            "name": "Softplus+Reg-FrozenPolicy",
            "head_class": SoftplusValueHead,
            "freeze_policy": True,
            "use_softplus": True,
            "use_regularization": True,
            "delta_aware": False
        },
        {
            "name": "DeltaAware-FrozenPolicy",
            "head_class": DeltaAwareValueHead,
            "freeze_policy": True,
            "use_softplus": False,
            "use_regularization": False,
            "delta_aware": True
        }
    ]
    
    # 存储所有变体的指标
    all_metrics = {}
    
    # 训练所有变体
    for variant in variants:
        print(f"\n===== 训练变体: {variant['name']} =====")
        value_head = variant["head_class"](hidden_size=hidden_dim)
        
        metrics = train_ledpo_variant(
            value_head=value_head,
            train_data=train_data,
            variant_name=variant["name"],
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            freeze_policy_model=variant["freeze_policy"],
            use_softplus=variant["use_softplus"],
            use_regularization=variant["use_regularization"],
            delta_aware=variant["delta_aware"]
        )
        
        all_metrics[variant["name"]] = metrics
    
    # 绘制结果对比
    plot_results(all_metrics, title="LEDPO不同变体对比")
    
    # 将结果保存为文本文件
    with open("results/ledpo_variants_results.txt", "w") as f:
        f.write("# LEDPO不同变体实验结果\n\n")
        
        for variant_name, metrics in all_metrics.items():
            f.write(f"## {variant_name}\n")
            f.write(f"最终beta_scale = {metrics['beta_scale'][-1]:.4f}\n")
            f.write(f"最终beta平均值 = {metrics['beta_mean'][-1]:.4f}\n")
            f.write(f"最终pos_beta = {metrics['positive_beta_avg'][-1]:.4f}\n")
            f.write(f"最终neg_beta = {metrics['negative_beta_avg'][-1]:.4f}\n")
            f.write(f"最终pos_beta/neg_beta比值 = {metrics['positive_beta_avg'][-1]/max(1e-5, metrics['negative_beta_avg'][-1]):.4f}\n\n")
    
    print("\n实验完成，请查看results目录中的结果")


if __name__ == "__main__":
    main() 