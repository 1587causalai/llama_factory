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
    # 当前版本必须提供 ppl 参数, 所以相关逻辑默认关闭
    use_ppl_approx: bool = False  # 是否使用神经网络近似 log(PPL) 

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

        # PPL 近似网络（后续版本考虑）
        pass

        self.epsilon = config.epsilon
        assert 0 <= self.epsilon <= 1, "epsilon must be in (0, 1)"

    def forward(self, hidden_states: torch.Tensor, cum_log_ppl: torch.Tensor) -> torch.Tensor:
        """计算动态 beta 值。
        
        Args:
            hidden_states: 最后一层隐状态, 形状为 [batch_size, seq_len, hidden_size]
            cum_log_ppl: 累积 log_ppl, 形状为 [batch_size, seq_len], 当前版本必须提供 cum_log_ppl 参数
        Returns:
            beta: 动态 beta 值, 形状为 [batch_size, seq_len]
        """
        device = hidden_states.device
        cum_log_ppl = cum_log_ppl.to(device) # [batch_size, seq_len]
        beta_scale = self.beta_scale.to(device) # [1]

        # 计算 f(x)
        f_x = 1 + self.epsilon * torch.tanh(self.context_adjust_net(hidden_states).squeeze(-1)) # [batch_size, seq_len]

        # 计算 log(PPL)
        if self.config.use_ppl_approx: # 后续版本考虑
            pass
        else:
            assert (cum_log_ppl >= 0).all(), "cum_log_ppl must be non-negative"

        return beta_scale * f_x * cum_log_ppl


class ExtendedModelWithBetaHead(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config # 获取 base_model 的 config, 方便后续使用
        beta_head_config = BetaHeadConfig(hidden_size=self.config.hidden_size)
        self.beta_head = BetaHead(beta_head_config)
        self.beta_head.to(base_model.device) # 确保 beta_head 和 base_model 在同一设备上

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """"
        前向传播函数。

        返回和 base_model 一致, 如果 base_model 返回是 dict, 就额外返回 beta_scale, beta, log_ppl. 当前版本我们直接返回 dict. 
        """
        # step 1: self.base_model forward
        outputs = self.base_model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True, return_dict=True, **kwargs)
        logits = outputs.logits # [batch_size, seq_len, vocab_size]
        hidden_states = outputs.hidden_states[-1] # [batch_size, seq_len, hidden_size], get last layer hidden states

        # step 2: compute cum_log_ppl with self.compute_cum_log_ppl 
        # for comput log ppl, 默认 labels 是 input_ids 的索引
        ppl_labels = input_ids.clone()
        ppl_labels[ppl_labels == self.base_model.config.pad_token_id] = -100
        cum_log_ppl = self.compute_cum_log_ppl(logits, labels=ppl_labels) # [batch_size, seq_len]  

        # step 3: compute dynamic beta through beta_head, 
        cum_dynamic_beta = self.beta_head(hidden_states, cum_log_ppl) # [batch_size, seq_len]
        beta_scale = self.beta_head.beta_scale.detach() # [1]
        
        # step 4: compute loss
        # 我记得 LLM forward , 当你传入参数非空 label 的时候, 会自动计算 loss
        
        # step 5: 返回结果
        outputs['cum_log_ppl'] = cum_log_ppl
        outputs['cum_dynamic_beta'] = cum_dynamic_beta
        outputs['beta_scale'] = beta_scale
        return outputs

    def compute_cum_log_ppl(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            """计算每个位置的累积 log_ppl。

            Args:
                logits (torch.Tensor): 模型的输出，形状为 [batch_size, seq_len, vocab_size]。
                labels (torch.Tensor): 真实的标签，形状为 [batch_size, seq_len]。

            Returns:
                torch.Tensor: 每个位置的累积 log_ppl，形状为 [batch_size, seq_len]。
            """
            # 计算交叉熵损失，忽略 padding token
            loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()  # 移除最后一个 token 的 logits
            shift_labels = labels[..., 1:].contiguous()      # 移除第一个 token 的 label
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())  # 恢复形状为 [batch_size, seq_len-1]

            # 创建 mask，标记有效 token
            mask = (shift_labels != -100).float() # [batch_size, seq_len-1]

            # 计算累积 NLL
            cum_nll = torch.cumsum(loss * mask, dim=1) # [batch_size, seq_len-1]

            # 计算累积有效 token 数量
            cum_lengths = torch.cumsum(mask, dim=1) # [batch_size, seq_len-1]

            # 计算累积 log_ppl = cum_nll / cum_lengths
            cum_log_ppl = torch.where(cum_lengths > 0, cum_nll / cum_lengths, torch.zeros_like(cum_nll)) # [batch_size, seq_len-1]

            # 补齐第一个位置（设为 0 是一种选择, 但是我们选择平均值填充）
            first_log_ppl = cum_log_ppl.mean(dim=1, keepdim=True) # [batch_size, 1]
            cum_log_ppl = torch.cat([first_log_ppl, cum_log_ppl], dim=1) # [batch_size, seq_len]

            # # 补齐最后一个位置（复制前一个位置的值）, 前面已经补齐第一个位置了, 再加上计算出来的 cum_log_ppl, 已经是 [batch_size, seq_len] 了.
            # last_log_ppl = cum_log_ppl[:, -1:].clone() # [batch_size, 1]
            # cum_log_ppl = torch.cat([cum_log_ppl, last_log_ppl], dim=1) # [batch_size, seq_len]

            return cum_log_ppl # [batch_size, seq_len]
        
    def idx_prompt_last_token(self, prompt_last_position: torch.Tensor, cum_log_ppl: torch.Tensor) -> torch.Tensor:
        """计算 prompt last token log_ppl from cum_log_ppl。
        
        从 cum_log_ppl 按照 index prompt_last_position 提取 log_ppl. 

        
        Args:
            prompt_last_position: 提示的最后一个位置 for each sample in batch, 形状为 [batch_size, 1]
            cum_log_ppl: 累积 log_ppl, 形状为 [batch_size, seq_len]
        Returns:
            torch.Tensor: 累积 log_ppl, 形状为 [batch_size, 1]
        """
        return torch.gather(cum_log_ppl, 1, prompt_last_position)
    

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("/root/models/Qwen1.5-0.5B-Chat")
    model = ExtendedModelWithBetaHead(model)
    print(model)

    # 创建示例输入
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)) # [batch_size, seq_len]
    attention_mask = torch.ones_like(input_ids) # [batch_size, seq_len] 
    outputs = model(input_ids, attention_mask=attention_mask)
    print(outputs['cum_dynamic_beta'])
    print(outputs['beta_scale'])
    assert outputs['cum_log_ppl'].shape == (batch_size, seq_len)
    assert outputs['cum_dynamic_beta'].shape == (batch_size, seq_len)
    assert outputs['beta_scale'].shape == (1,)

    cum_log_ppl = model.compute_cum_log_ppl(outputs.logits, input_ids)
    print(cum_log_ppl)
    assert cum_log_ppl.shape == (batch_size, seq_len)

    # 每个样本的 prompt token 最后一个位置索引
    prompt_last_position = torch.randint(0, seq_len, (batch_size, 1))
    prompt_log_ppl = model.idx_prompt_last_token(prompt_last_position, cum_log_ppl)
    print(prompt_log_ppl)
    assert prompt_log_ppl.shape == (batch_size, 1)

    