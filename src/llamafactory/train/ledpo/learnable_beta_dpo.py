import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class BetaHeadConfig:
    """BetaHead 的配置类"""
    hidden_size: int = 768  # 输入的隐状态维度
    context_net_type: str = "linear"  # 上下文调整网络类型: "linear" 或 "mlp"
    context_mlp_hidden_size: int = 128  # 上下文 MLP 中间层维度
    epsilon: float = 0.0  # 控制 f(x) 范围 [1-ε, 1+ε]，建议 [0.05, 0.2]
    dropout: float = 0.0  # Dropout 概率
    w_trainable: bool = True  # 是否训练 beta_scale
    w_init: float = 0.01  # beta_scale 初始值，建议 [0.01, 1.0]
    use_ppl_approx: bool = False  # 是否使用神经网络近似 log(PPL)
    ppl_net_type: str = "linear"  # PPL 近似网络类型
    ppl_mlp_hidden_size: int = 128  # PPL 近似网络中间层维度

class BetaHead(nn.Module):
    """动态 beta 值计算的头部网络，用于 DPO 算法中的自适应权衡。

    计算公式: β(x) = beta_scale * log(PPL(x)) * f(x)
    - beta_scale: 可学习或固定的缩放参数
    - log(PPL(x)): 输入 x 的困惑度对数，可通过 ppl 或神经网络近似计算
    - f(x) = 1 + epsilon * tanh(NN(hidden_states)): 基于上下文的调整函数
    """
    def __init__(self, config: BetaHeadConfig):
        super().__init__()
        self.config = config

        # 初始化 beta_scale
        if config.w_trainable:
            self.beta_scale = nn.Parameter(torch.ones(1) * config.w_init)
        else:
            self.register_buffer("beta_scale", torch.ones(1) * config.w_init)

        # 上下文调整网络
        if config.context_net_type == "linear":
            self.context_adjust_net = nn.Linear(config.hidden_size, 1)
        elif config.context_net_type == "mlp":
            self.context_adjust_net = nn.Sequential(
                nn.Linear(config.hidden_size, config.context_mlp_hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.context_mlp_hidden_size, 1)
            )
        else:
            raise ValueError(f"Unknown context_net_type: {config.context_net_type}")

        # PPL 近似网络（可选）
        self.use_ppl_approx = config.use_ppl_approx
        if self.use_ppl_approx:
            if config.ppl_net_type == "linear":
                self.ppl_approx_net = nn.Linear(config.hidden_size, 1)
            elif config.ppl_net_type == "mlp":
                self.ppl_approx_net = nn.Sequential(
                    nn.Linear(config.hidden_size, config.ppl_mlp_hidden_size),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.ppl_mlp_hidden_size, 1)
                )
            else:
                raise ValueError(f"Unknown ppl_net_type: {config.ppl_net_type}")

        self.epsilon = config.epsilon
        assert 0 < self.epsilon < 1, "epsilon must be in (0, 1)"

    def forward(self, hidden_states: torch.Tensor, ppl: torch.Tensor) -> torch.Tensor:
        """计算动态 beta 值。"""
        device = hidden_states.device
        ppl = ppl.to(device)
        beta_scale = self.beta_scale.to(device)

        # 计算 f(x)
        f_x = 1 + self.epsilon * torch.tanh(self.context_adjust_net(hidden_states).squeeze(-1))

        # 计算 log(PPL)
        if self.use_ppl_approx:
            log_ppl = self.ppl_approx_net(hidden_states).squeeze(-1)
        else:
            assert (ppl > 0).all(), "ppl must be positive"
            log_ppl = torch.log(ppl)

        return beta_scale * f_x * log_ppl