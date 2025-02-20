# DPO 数据处理机制与 PPL 计算分析

## 1. DPO 数据结构

### 1.1 原始数据格式
对于每个训练样本，包含三个部分：
- prompt (x): 输入提示
- chosen response (y_w): 期望的回答
- rejected response (y_l): 不期望的回答

### 1.2 数据处理流程

1. **数据编码阶段** (`PairwiseDatasetProcessor._encode_data_example`):
```python
# 编码 chosen 和 rejected 样本
prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, chosen_messages)
_, rejected_ids = template.encode_oneturn(tokenizer, rejected_messages)

# 构造 labels
chosen_labels = [IGNORE_INDEX] * len(prompt_ids) + chosen_ids
rejected_labels = [IGNORE_INDEX] * len(prompt_ids) + rejected_ids
```

2. **数据拼接阶段** (`PairwiseDataCollatorWithPadding.__call__`):
```python
# 将一个 batch 的数据拼接为:
concatenated_features = []
for feature in features:
    # chosen samples
    concatenated_features.append({
        "input_ids": feature["chosen_input_ids"],      # [prompt + chosen]
        "attention_mask": feature["chosen_attention_mask"],
        "labels": feature["chosen_labels"]             # [IGNORE_INDEX * prompt_len + chosen]
    })
    # rejected samples
    concatenated_features.append({
        "input_ids": feature["rejected_input_ids"],    # [prompt + rejected]
        "attention_mask": feature["rejected_attention_mask"],
        "labels": feature["rejected_labels"]           # [IGNORE_INDEX * prompt_len + rejected]
    })
```

### 1.3 实际数据结构示例

基于实际测试结果，我们以一个具体的样本为例：

```
Chosen 样本:
- 总长度: 379 tokens
- IGNORE_INDEX 比例: 11.08%
- 有效标签数: 337 tokens

Rejected 样本:
- 总长度: 159 tokens
- IGNORE_INDEX 比例: 26.42%
- 有效标签数: 117 tokens
```

注意事项：
1. 标签序列中的 IGNORE_INDEX (-100) 不仅出现在 prompt 部分，也可能出现在序列末尾的特殊标记处
2. Response 部分可能包含特殊标记（如 `<|endoftext|>`），这些标记的标签也是 IGNORE_INDEX
3. 数据集格式需要在 `dataset_info.json` 中正确配置，例如：
```json
{
  "dpo_en_demo": {
    "file_name": "dpo_en_demo.json",
    "ranking": true,
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
}
```

## 2. PPL 计算机制

### 2.1 理论基础

根据 Learnable Beta DPO 论文，对于一对偏好数据 (x, y_w, y_l)，β(x) 的计算只应该考虑输入 prompt x 的 PPL：

$$\mathrm{PPL}_{\pi_\theta}(x) = \exp \left( - \frac{1}{m} \sum_{i=1}^m \log \pi_\theta(x_i | x_{<i}) \right)$$

其中 m 是 prompt 的长度。

### 2.2 实现注意事项

1. **Prompt 边界识别**：
   - 不能仅依赖 `labels != IGNORE_INDEX` 来判断 prompt 结束位置
   - 应该使用 `<|im_start|>assistant` 的位置作为分界点
   - 或者使用模板提供的 `encode_oneturn` 返回的 prompt_ids 长度

2. **Context Embedding 提取**：
   - 当前使用 `hidden_states[:, -1, :]` 可能包含了 response 信息
   - 应该使用 prompt 结束位置的 hidden state：
   ```python
   prompt_end_indices = torch.tensor(prompt_lengths) - 1
   batch_indices = torch.arange(batch_size)
   context_embedding = hidden_states[batch_indices, prompt_end_indices, :]
   ```

3. **PPL 计算范围**：
   - 只使用 prompt 部分的 token 计算 PPL
   - 忽略特殊标记（如 `<|endoftext|>`）的影响
   - 使用 attention_mask 确保只计算有效 token

### 2.3 建议的实现

```python
def get_dynamic_beta_and_ppl(self, input_ids, attention_mask, model_outputs=None):
    # 1. 获取模型输出
    outputs = model_outputs or self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    
    # 2. 找到 assistant 标记的位置作为 prompt 结束位置
    assistant_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>assistant")
    batch_size = input_ids.size(0)
    prompt_lengths = []
    for i in range(batch_size):
        assistant_pos = (input_ids[i] == assistant_token_id).nonzero()
        if len(assistant_pos) > 0:
            prompt_lengths.append(assistant_pos[-1].item())
        else:
            prompt_lengths.append(input_ids.size(1))
    
    # 3. 提取正确的 context embedding
    prompt_end_indices = torch.tensor(prompt_lengths) - 1
    batch_indices = torch.arange(batch_size)
    context_embedding = outputs.hidden_states[-1][batch_indices, prompt_end_indices, :]
    
    # 4. 计算 prompt 部分的 PPL
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:].float()
    
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction='none'
    ).reshape_as(labels)
    
    prompt_ppls = []
    for i in range(batch_size):
        prompt_len = prompt_lengths[i] - 1
        if prompt_len > 0:
            prompt_loss = loss[i, :prompt_len]
            prompt_mask = mask[i, :prompt_len]
            avg_prompt_loss = (prompt_loss * prompt_mask).sum() / prompt_mask.sum()
            prompt_ppls.append(torch.exp(avg_prompt_loss))
    
    ppl = torch.stack(prompt_ppls)
    
    # 5. 计算动态 beta
    beta = self.beta_head.get_dynamic_beta(context_embedding, ppl)
    
    return beta, ppl
```

## 3. 验证方案

### 3.1 单元测试

1. **数据处理测试**：
   - 验证 prompt 和 response 的正确分割
   - 验证标签序列的正确性
   - 验证特殊标记的处理

2. **PPL 计算测试**：
   - 使用简单的已知序列
   - 手动计算预期的 PPL
   - 验证实际计算结果

3. **Context Embedding 测试**：
   - 验证提取的 embedding 确实来自 prompt 结束位置
   - 验证 embedding 不包含 response 信息

### 3.2 集成测试

1. **完整流程测试**：
   - 从原始数据到最终的 beta 值
   - 验证 beta 值的合理性
   - 验证梯度流动

2. **边界情况测试**：
   - 特别长的序列
   - 包含多个 assistant 标记的序列
   - 没有 assistant 标记的序列

## 4. 下一步行动

1. 实现上述建议的修改
2. 添加完整的测试套件
3. 更新文档和注释
4. 添加运行时检查和断言 