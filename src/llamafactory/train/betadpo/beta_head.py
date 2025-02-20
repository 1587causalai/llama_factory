import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from dataclasses import dataclass

"""
重要说明：
1. 对于偏好数据集中的一对回复(chosen 和 rejected)，我们只需要计算一次 beta 值，
   这个 beta 值是基于输入文本(prompt)计算的，而不是基于生成的回复。

2. 计算流程说明：
   - 输入文本 x 通过策略模型得到上下文表示 h_π_θ(x)
   - 同时计算输入文本的困惑度 PPL_π_θ(x)
   - beta 值的计算公式为：β(x) = w · log(PPL(x)) · f(x)
   其中 f(x) = 1 + ε·tanh(NN(h_π_θ(x))) 是基于上下文的调整函数

3. 注意事项：
   - 困惑度计算仅基于输入文本，与生成的回复无关
   - 对于同一个输入文本的不同回复，使用相同的 beta 值
   - beta 值反映了当前输入的难度和上下文特征

4. 实现细节：
   - forward() 方法返回 w * f(x) 部分
   - get_dynamic_beta() 方法返回完整的 beta 值，包含 PPL 和 sigmoid 激活
"""

@dataclass
class BetaHeadConfig:
    """BetaHead 的配置类"""
    hidden_size: int = 768  # 输入的隐状态维度
    nn_type: str = "linear"  # 神经网络类型: "linear" 或 "mlp"
    mlp_hidden_size: int = 128  # MLP 中间层维度(仅在 nn_type="mlp" 时使用)
    epsilon: float = 0.1  # 控制调整函数 f(h) 的范围 [1-ε, 1+ε]
    dropout: float = 0.1  # dropout 概率

class BetaHead(nn.Module):
    """动态 beta 值计算的头部网络

    计算公式: β(x) = w · PPL(x) · f(x)
    其中:
    - w 是可学习的参数 (现在移入 BetaHead 内部)
    - PPL(x) 是策略模型计算的困惑度
    - f(x) = 1 + ε·tanh(NN(h_π_θ(x))) 是基于上下文的调整函数
    """

    def __init__(self, config: BetaHeadConfig):
        super().__init__()
        self.config = config

        # 初始化可学习参数 w (移入 BetaHead 内部)
        self.register_parameter("w", nn.Parameter(torch.ones(1)* 0.01))  # 使用 register_parameter 确保参数被正确注册

        # 根据配置构建神经网络 NN(x)
        if config.nn_type == "linear":
            self.nn = nn.Sequential(
                nn.Linear(config.hidden_size, 1),
                nn.Dropout(config.dropout)
            )
        elif config.nn_type == "mlp":
            self.nn = nn.Sequential(
                nn.Linear(config.hidden_size, config.mlp_hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.mlp_hidden_size, 1),
                nn.Dropout(config.dropout)
            )
        else:
            raise ValueError(f"Unknown nn_type: {config.nn_type}")

        self.epsilon = config.epsilon

    def forward(
        self,
        context_embedding: torch.Tensor,  # shape: [batch_size, hidden_size]
        ppl: torch.Tensor,  # shape: [batch_size]
    ) -> torch.Tensor:  # shape: [batch_size]
        """计算动态 beta 值

        Args:
            context_embedding: 策略模型最后一层隐状态 h_π_θ(x)
            ppl: 策略模型计算的困惑度 PPL_π_θ(x)

        Returns:
            beta: 动态 beta 值 β(x; π_θ)  (现在直接返回  w * PPL(x) * f(x) 中的  w * f(x) 部分)
        """
        # 确保所有张量在同一设备上
        device = context_embedding.device
        ppl = ppl.to(device)
        w = self.w.to(device)
        
        # 1. 计算调整函数 f(x) = 1 + ε·tanh(NN(h))
        nn_output = self.nn(context_embedding).squeeze(-1)  # [batch_size]
        f_x = 1 + self.epsilon * torch.tanh(nn_output)

        return w * f_x * torch.log(ppl)

    def extra_repr(self) -> str:
        """返回额外的表示信息"""
        return f"nn_type={self.config.nn_type}, epsilon={self.epsilon}"