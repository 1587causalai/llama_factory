import torch
import torch.nn as nn
import torch.nn.functional as F

class LengthBasedBetaHead(nn.Module):
    """基于输入长度计算 beta 值的模型"""
    
    def __init__(self, beta_base=0.1, hidden_size=64, min_beta=0.01, max_beta=10.0):
        """
        初始化基于长度的 beta head
        
        Args:
            beta_base: 基准 beta 值，用于初始化
            hidden_size: 隐藏层大小
            min_beta: beta 的最小值
            max_beta: beta 的最大值
        """
        super().__init__()
        self.beta_base = beta_base
        self.min_beta = min_beta
        self.max_beta = max_beta
        
        # 简单的两层神经网络
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus()  # 确保输出非负
        )
        
        # 初始化为产生接近 beta_base 的值
        with torch.no_grad():
            # 假设标准化后的长度为 0.5 (500个token)
            self.net[2].bias.fill_(torch.log(torch.exp(torch.tensor(beta_base)) - 1.0))
        
    def forward(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        计算基于输入长度的 beta 值
        
        Args:
            lengths: 形状为 [batch_size] 的张量，表示每个样本的输入长度
            
        Returns:
            形状为 [batch_size] 的张量，表示每个样本的 beta 值
        """
        # 归一化长度
        normalized_lengths = lengths.float().unsqueeze(1) / 1000.0
        
        # 计算 beta 值
        beta = self.net(normalized_lengths).squeeze(1)
        
        # 限制 beta 值范围
        beta = torch.clamp(beta, min=self.min_beta, max=self.max_beta)
        
        return beta 

class HiddenStateBetaHead(nn.Module):
    """基于提示的最后一个token的隐藏状态计算beta值的模型"""
    
    def __init__(
        self, 
        hidden_size: int,  # 模型隐藏状态的维度 
        beta_base: float = 0.1,  # 基准beta值
        projection_dim: int = 64,  # 投影维度
        min_beta: float = 0.01,  # beta的最小值
        max_beta: float = 100.0,  # beta的最大值
        use_layernorm: bool = False  # 是否使用层标准化
    ):
        """
        初始化基于隐藏状态的beta head
        
        Args:
            hidden_size: 输入隐藏状态的维度
            beta_base: 基准beta值，用于初始化
            projection_dim: 投影层的维度
            min_beta: beta的最小值
            max_beta: beta的最大值
            use_layernorm: 是否使用层标准化
        """
        super().__init__()
        self.beta_base = beta_base
        self.min_beta = min_beta
        self.max_beta = max_beta
        
        layers = []
        
        # 可选的层标准化
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_size))
            
        # 投影层+两层神经网络
        layers.extend([
            nn.Linear(hidden_size, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Linear(projection_dim // 2, 1),
            nn.Softplus()  # 确保输出非负
        ])
        
        self.net = nn.Sequential(*layers)
        
        # 初始化为产生接近beta_base的值
        with torch.no_grad():
            self.net[-2].bias.fill_(torch.log(torch.exp(torch.tensor(beta_base)) - 1.0))
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        计算基于隐藏状态的beta值
        
        Args:
            hidden_states: 形状为[batch_size, hidden_size]的张量，表示每个样本提示最后一个token的隐藏状态
            
        Returns:
            形状为[batch_size]的张量，表示每个样本的beta值
        """
        # 计算beta值
        beta = self.net(hidden_states).squeeze(1)
        
        # 限制beta值范围
        beta = torch.clamp(beta, min=self.min_beta, max=self.max_beta)
        
        return beta 