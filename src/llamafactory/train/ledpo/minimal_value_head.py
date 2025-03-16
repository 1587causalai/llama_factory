import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalValueHead(nn.Module):
    """
    最小可行的ValueHead实现，用于预测每个样本的beta值
    简化了网络结构，只使用单层线性变换和直接激活函数
    """
    
    def __init__(self, hidden_size: int, beta_min: float = 0.001, beta_max: float = 100.0):
        super().__init__()
        # 简单的线性层
        self.linear = nn.Linear(hidden_size, 1)
        
        # beta参数范围
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """简单的权重初始化"""
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear.bias, 1.0)  # 稍微大点的偏置使初始输出倾向于更大的beta值
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播函数"""
        # 获取原始输出并应用ReLU确保非负
        raw_output = F.relu(self.linear(hidden_states)).squeeze(-1)
        
        # 直接映射到beta范围，使用softplus确保平滑过渡和正值
        beta = self.beta_min + F.softplus(raw_output)
        
        return beta


class SimpleValueHead(nn.Module):
    """
    简单的ValueHead实现，带有beta_scale参数但结构更简单
    """
    
    def __init__(self, hidden_size: int, beta_min: float = 0.001, beta_max: float = 100.0, init_beta_scale: float = 5.0):
        super().__init__()
        # 简单的线性层
        self.linear = nn.Linear(hidden_size, 1)
        
        # 可学习的beta_scale参数
        self.beta_scale = nn.Parameter(torch.tensor(init_beta_scale), requires_grad=True)
        
        # beta参数范围
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """简单的权重初始化"""
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear.bias, 0.5)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播函数"""
        # 获取原始输出并应用Sigmoid将值压缩到0-1范围
        raw_output = torch.sigmoid(self.linear(hidden_states)).squeeze(-1)
        
        # 应用beta_scale (确保为正值)
        beta_scale_positive = F.softplus(self.beta_scale)
        
        # 映射到beta范围
        beta_range = self.beta_max - self.beta_min
        beta = self.beta_min + raw_output * beta_range * (beta_scale_positive / 10.0)
        
        return beta


# 使用示例
def test_value_heads():
    """测试不同的ValueHead实现"""
    hidden_size = 768
    batch_size = 4
    seq_len = 1
    
    # 创建模拟输入
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 测试MinimalValueHead
    minimal_head = MinimalValueHead(hidden_size)
    minimal_beta = minimal_head(hidden_states)
    print(f"MinimalValueHead output shape: {minimal_beta.shape}")
    print(f"MinimalValueHead output range: {minimal_beta.min().item():.4f} to {minimal_beta.max().item():.4f}")
    
    # 测试SimpleValueHead
    simple_head = SimpleValueHead(hidden_size)
    simple_beta = simple_head(hidden_states)
    print(f"SimpleValueHead output shape: {simple_beta.shape}")
    print(f"SimpleValueHead output range: {simple_beta.min().item():.4f} to {simple_beta.max().item():.4f}")
    print(f"SimpleValueHead beta_scale: {F.softplus(simple_head.beta_scale).item():.4f}")


if __name__ == "__main__":
    test_value_heads() 