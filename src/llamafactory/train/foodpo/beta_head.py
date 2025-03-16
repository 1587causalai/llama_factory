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