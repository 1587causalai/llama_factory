#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LEDPO动态beta趋零问题修复方案
=========================

这个脚本提供了修复LEDPO动态beta趋零问题的完整解决方案。
根据分析，当使用freeze_policy_model=True时，模型梯度无法正常传递，
导致value_head的beta_scale参数不断减小，使beta值趋近于零。

以下修复方案包括：
1. 使用F.softplus确保beta_scale始终为正值
2. 添加额外的正则化项，防止beta_scale值过小
3. 增加beta_scale的监控代码

作者: [您的名字]
日期: [日期]
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Literal
import logging

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ====================================
# 一、修改ValueHead的forward方法
# ====================================
"""
原理：使用F.softplus确保beta_scale始终为正值，
并提供更多调试信息来监控beta_scale和beta值的变化。
"""

class ValueHead(nn.Module):
    """
    简单的value head网络，用于预测每个样本的beta值
    修复版本: 使用softplus确保beta_scale始终为正
    """
    
    def __init__(self, hidden_size: int, beta_min: float = 0.01, beta_max: float = 100.0):
        super().__init__()
        # 初始化beta_scale为10.0，这是一个全局缩放因子
        self.beta_scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        
        # 构建更简单的value head网络
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),  # 使用GELU激活函数代替ReLU
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()  # 确保输出为正值
        )   
        
        # 初始化网络权重
        self.value_head.apply(self.init_weights)
        
        # 设置beta的最小值和最大值
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # 记录初始化信息
        logger.info(f"ValueHead initialized with beta_scale={self.beta_scale.item():.4f}, beta_min={beta_min}, beta_max={beta_max}")
        
    def init_weights(self, module):
        """初始化网络权重，使用较大的标准差以避免梯度消失"""
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为0.1
            nn.init.normal_(module.weight, mean=0.0, std=0.1)
            # 将偏置初始化为小正数，而不是0，以避免初始输出过小
            nn.init.constant_(module.bias, 0.1)
            
            logger.debug(f"Initialized Linear layer: weight_shape={module.weight.shape}, bias_shape={module.bias.shape}")
            logger.debug(f"Weight stats: mean={module.weight.mean().item():.4f}, std={module.weight.std().item():.4f}")
            logger.debug(f"Bias stats: mean={module.bias.mean().item():.4f}, std={module.bias.std().item():.4f}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数 - 修复版本
        
        修复内容:
        1. 使用F.softplus确保beta_scale始终为正值
        2. 增加调试信息记录
        
        Args:
            hidden_states: 最后一层hidden states，形状为[batch_size, hidden_size]
            
        Returns:
            beta值，形状为[batch_size]
        """
        # 输入是最后一层hidden states
        raw_beta = self.value_head(hidden_states)
        
        # ===== 修复点1: 应用softplus确保beta_scale始终为正 =====
        # 用softplus确保beta_scale始终为正，起点为10.0
        beta_scale_positive = F.softplus(self.beta_scale)
        # 使用beta_scale_positive作为beta值的缩放因子
        scaled_beta = beta_scale_positive * raw_beta
        
        # 将输出截断到[beta_min, beta_max]范围
        clamped_beta = torch.clamp(scaled_beta, min=self.beta_min, max=self.beta_max)
        
        # ===== 修复点2: 增加更多调试信息 =====
        if torch.rand(1).item() < 0.1:  # 增加打印概率到10%
            logger.info(f"ValueHead forward - raw_beta: min={raw_beta.min().item():.4f}, max={raw_beta.max().item():.4f}, mean={raw_beta.mean().item():.4f}")
            logger.info(f"ValueHead forward - beta_scale (raw): {self.beta_scale.item():.4f}")
            logger.info(f"ValueHead forward - beta_scale (positive): {beta_scale_positive.item():.4f}")
            logger.info(f"ValueHead forward - scaled_beta: min={scaled_beta.min().item():.4f}, max={scaled_beta.max().item():.4f}, mean={scaled_beta.mean().item():.4f}")
            logger.info(f"ValueHead forward - clamped_beta: min={clamped_beta.min().item():.4f}, max={clamped_beta.max().item():.4f}, mean={clamped_beta.mean().item():.4f}")
        
        return clamped_beta


# ====================================
# 二、添加beta_scale正则化损失
# ====================================
"""
原理：在compute_preference_loss中添加一个额外的正则化损失，
使得当delta>0时鼓励增大beta，当delta<0时允许减小beta，
但不会让beta_scale过分减小至接近零。
"""

def compute_preference_loss_fixed(
    self,
    policy_chosen_logps: "torch.Tensor",
    policy_rejected_logps: "torch.Tensor",
    reference_chosen_logps: Optional["torch.Tensor"],
    reference_rejected_logps: Optional["torch.Tensor"],
    chosen_prompt_last_token_hidden: Optional["torch.Tensor"] = None,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
    """
    使用动态beta计算preference loss的修复版本
    
    修复内容:
    1. 添加beta_scale正则化，防止beta值过小
    2. 增加更多调试信息
    """
    # 初始化dynamic_beta和beta
    dynamic_beta = self.beta  # 默认使用固定beta
    beta_reg_loss = torch.tensor(0.0, device=policy_chosen_logps.device)  # 初始化正则化损失
    
    # 如果使用动态beta且提供了hidden states
    if self.use_dynamic_beta and chosen_prompt_last_token_hidden is not None:
        # 计算dynamic_beta
        dynamic_beta = self.value_head(chosen_prompt_last_token_hidden).squeeze(-1)
        
        # 记录当前beta_scale值
        # ===== 修复点1: 使用softplus获取实际的beta_scale =====
        beta_scale_positive = F.softplus(self.value_head.beta_scale)
        logger.info(f"ValueHead beta_scale (raw): {self.value_head.beta_scale.item():.4f}")
        logger.info(f"ValueHead beta_scale (positive): {beta_scale_positive.item():.4f}")
        logger.info(f"Dynamic beta stats: min={dynamic_beta.min().item():.4f}, max={dynamic_beta.max().item():.4f}, mean={dynamic_beta.mean().item():.4f}")
        
    # 计算delta值(如果使用参考模型)
    if reference_chosen_logps is not None and reference_rejected_logps is not None:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        delta = pi_logratios - ref_logratios
        
        # 添加调试信息
        positive_delta_mask = (delta > 0)
        negative_delta_mask = (delta < 0)
        pos_count = positive_delta_mask.sum().item()
        neg_count = negative_delta_mask.sum().item()
        
        logger.info(f"Delta stats: min={delta.min().item():.4f}, max={delta.max().item():.4f}, mean={delta.mean().item():.4f}")
        logger.info(f"Delta distribution: positive={pos_count}, negative={neg_count}, ratio={pos_count/(pos_count+neg_count+1e-10):.4f}")
        
        if self.use_dynamic_beta:
            # 计算正负delta样本对应的beta平均值
            positive_beta_avg = dynamic_beta[positive_delta_mask].mean().item() if pos_count > 0 else 0
            negative_beta_avg = dynamic_beta[negative_delta_mask].mean().item() if neg_count > 0 else 0
            logger.info(f"Beta for positive delta: {positive_beta_avg:.4f}, Beta for negative delta: {negative_beta_avg:.4f}")
            
            # ===== 修复点2: 添加beta_scale正则化逻辑 =====
            # 防止使用freeze_policy_model时beta_scale不断减小
            if self.freeze_policy_model and hasattr(self.value_head, "beta_scale"):
                # 计算正则化损失: 鼓励beta_scale不要过小，保持在合理范围内
                # 当beta_scale低于某个阈值(例如1.0)时，增加惩罚
                beta_scale_threshold = 1.0
                if self.value_head.beta_scale.item() < beta_scale_threshold:
                    beta_reg_factor = 0.1  # 正则化系数
                    beta_reg_loss = beta_reg_factor * F.relu(beta_scale_threshold - self.value_head.beta_scale)
                    logger.info(f"Adding beta_scale regularization: {beta_reg_loss.item():.4f}")
    
    # 根据使用的方法计算损失
    if not self.finetuning_args.use_ref_model:
        if self.loss_type == "orpo":
            losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps, dynamic_beta)
        elif self.loss_type == "simpo":
            losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps, dynamic_beta)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

        chosen_rewards = dynamic_beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = dynamic_beta * policy_rejected_logps.to(self.accelerator.device).detach()
    else:
        # 根据use_disco参数决定使用哪种损失计算方法
        if self.use_disco:
            # 使用DISCO损失计算方法
            losses, chosen_rewards, rejected_rewards = self.disco_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, dynamic_beta
            )
        else:
            # 使用标准DPO损失计算方法
            losses, chosen_rewards, rejected_rewards = self.dpo_loss_with_dynamic_beta(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, dynamic_beta
            )
    
    # ===== 修复点3: 添加正则化损失到总损失 =====
    if beta_reg_loss.item() > 0:
        losses = losses + beta_reg_loss
        logger.info(f"Total loss after beta regularization: {losses.mean().item():.4f}")

    return losses, chosen_rewards, rejected_rewards, dynamic_beta


# ====================================
# 三、更新优化器创建函数
# ====================================
"""
原理：为value_head参数设置更高的初始学习率，
并确保beta_scale有足够大的初始化值，让其不易衰减到零。
"""

def create_optimizer_fixed(self) -> "torch.optim.Optimizer":
    """
    创建优化器的修复版本
    
    修复内容:
    1. 为value_head参数设置更高的学习率
    2. 确保beta_scale参数的正确初始化
    """
    if self.optimizer is None:
        # 创建基本优化器
        self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)

        if self.optimizer is None:
            self.optimizer = super().create_optimizer()
            
        # 添加value_head参数到优化器
        if self.use_dynamic_beta and hasattr(self, "value_head"):
            # 确保value_head在正确设备上
            if hasattr(self.model, "device"):
                self.value_head = self.value_head.to(self.model.device)
            
            # 获取value_head参数
            value_head_params = list(self.value_head.parameters())
            
            # 打印value_head参数信息
            logger.info("ValueHead parameters:")
            for name, param in self.value_head.named_parameters():
                logger.info(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
            
            # ===== 修复点1: 为value_head参数设置更高的学习率 =====
            # 为beta_scale设置更高的学习率
            value_head_lr = self.args.learning_rate * 20.0  # 增加到原来的20倍
            
            # 添加参数组
            params_config = {
                "params": value_head_params,
                "lr": value_head_lr,  # 使用更高的学习率
                "weight_decay": 0.0,  # 减少权重衰减，防止beta_scale变小
            }
            
            # 复制原优化器配置（除了学习率和参数）
            for k, v in self.optimizer.param_groups[0].items():
                if k != "params" and k != "lr" and k != "weight_decay":
                    params_config[k] = v
            
            # 添加参数组
            self.optimizer.add_param_group(params_config)
            
            # 打印优化器信息
            logger.info("Optimizer param groups:")
            for i, group in enumerate(self.optimizer.param_groups):
                logger.info(f"  Group {i}: {len(group['params'])} parameters, lr={group['lr']}, weight_decay={group.get('weight_decay', 'N/A')}")
            
            logger.info(f"ValueHead parameters added to optimizer with lr={value_head_lr}")
            logger.info(f"beta_scale initial value: {self.value_head.beta_scale.item():.4f}")
            
            # ===== 修复点2: 确保beta_scale有足够大的初始值 =====
            # 如果beta_scale初始值太小，重新设置一个更大的值
            with torch.no_grad():
                if self.value_head.beta_scale.item() < 5.0:
                    self.value_head.beta_scale.fill_(10.0)
                    logger.info(f"Reset beta_scale to {self.value_head.beta_scale.item():.4f}")

    return self.optimizer


# ====================================
# 四、修复实施步骤
# ====================================
"""
以下是实施上述修复的步骤：

1. 备份原文件:
   cp src/llamafactory/train/ledpo/trainer.py src/llamafactory/train/ledpo/trainer.py.bak

2. 替换ValueHead类的forward方法:
   在src/llamafactory/train/ledpo/trainer.py中找到ValueHead类的forward方法，
   替换为上面提供的版本。

3. 替换compute_preference_loss方法:
   在LEDPOTrainer类中找到compute_preference_loss方法，
   替换为上面提供的compute_preference_loss_fixed版本。

4. 替换create_optimizer方法:
   在LEDPOTrainer类中找到create_optimizer方法，
   替换为上面提供的create_optimizer_fixed版本。

5. 重新进行训练，监控beta_scale的变化。
"""


# ====================================
# 五、完整修复注入代码
# ====================================
def inject_fixes():
    """
    这个函数将修复注入到现有的LEDPO代码中。
    实际使用时，请直接修改原始文件，而不是使用这种方式。
    """
    import importlib
    import types
    
    try:
        # 导入trainer模块
        from llamafactory.train.ledpo import trainer
        
        # 替换ValueHead.forward方法
        original_forward = trainer.ValueHead.forward
        trainer.ValueHead.forward = types.MethodType(ValueHead.forward, trainer.ValueHead)
        
        # 替换compute_preference_loss方法
        original_compute_preference_loss = trainer.LEDPOTrainer.compute_preference_loss
        trainer.LEDPOTrainer.compute_preference_loss = types.MethodType(compute_preference_loss_fixed, trainer.LEDPOTrainer)
        
        # 替换create_optimizer方法
        original_create_optimizer = trainer.LEDPOTrainer.create_optimizer
        trainer.LEDPOTrainer.create_optimizer = types.MethodType(create_optimizer_fixed, trainer.LEDPOTrainer)
        
        logger.info("成功注入LEDPO修复!")
        return True
    except Exception as e:
        logger.error(f"注入LEDPO修复失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    # 这部分代码仅用于演示，实际使用时请直接修改源代码文件
    print("LEDPO动态beta修复工具")
    print("===================================")
    print("这个脚本提供了修复LEDPO动态beta趋零问题的解决方案。")
    print("请按照脚本中的'修复实施步骤'进行操作，直接修改源代码文件。")
    print("===================================") 