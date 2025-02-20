import os
import sys
import torch

# 添加项目根目录到 Python 路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# 在设置好路径后再导入
from src.llamafactory.train.betadpo.beta_head import BetaHead, BetaHeadConfig

def test_beta_head():
    # 1. 创建配置和模型
    config = BetaHeadConfig(
        hidden_size=768,
        nn_type="mlp",
        epsilon=0.1
    )
    model = BetaHead(config)
    
    # 2. 准备输入数据
    batch_size = 2
    context_embedding = torch.randn(batch_size, 768)  # 模拟隐状态
    ppl = torch.tensor([1.5, 2.0])  # 模拟困惑度值
    
    # 3. 前向计算
    beta = model(context_embedding, ppl)
    
    # 4. 打印结果
    print("\nTest BetaHead output:")
    print(f"Input PPL: {ppl}")
    print(f"Output beta: {beta}")
    print(f"Beta range: [{beta.min():.3f}, {beta.max():.3f}]")
    
    # 5. 基本检查
    assert beta.shape == (batch_size,), "Wrong output shape"
    assert torch.all(beta > 0), "Beta values should be positive"
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_beta_head() 