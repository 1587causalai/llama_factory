# SFT (Supervised Fine-Tuning) 前向传播分析

本文详细分析LLaMA Factory中SFT算法的前向传播流程，重点关注数据流和计算机制。

## 1. 算法原理概述

监督式微调(Supervised Fine-Tuning, SFT)是最基础的预训练语言模型微调方法，通过有标签的输入-输出对，以监督学习的方式调整模型参数。SFT通常是其他高级微调方法（如DPO、PPO等）的基础，也是从预训练到特定任务适配的首要步骤。

在LLaMA Factory中，SFT实现为seq2seq式的训练范式，使用自回归语言建模目标，即预测序列中的下一个token。

## 2. 数据流和输入格式

### 2.1 输入数据格式

SFT训练需要的数据通常是成对的输入-输出样本，格式为：
- 输入(prompt): 用户指令或上下文
- 输出(response): 期望模型生成的回答

### 2.2 数据处理流程

1. **数据加载和预处理**：
   ```python
   tokenizer_module = load_tokenizer(model_args)
   tokenizer = tokenizer_module["tokenizer"]
   template = get_template_and_fix_tokenizer(tokenizer, data_args)
   dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
   ```

2. **数据整理器(Collator)配置**：
   ```python
   data_collator = SFTDataCollatorWith4DAttentionMask(
       template=template,
       model=model if not training_args.predict_with_generate else None,
       pad_to_multiple_of=8 if training_args.do_train else None,
       label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
       block_diag_attn=model_args.block_diag_attn,
       attn_implementation=getattr(model.config, "_attn_implementation", None),
       compute_dtype=model_args.compute_dtype,
       **tokenizer_module,
   )
   ```

3. **输入批次格式**：
   最终输入模型的批次包含以下关键字段：
   - `input_ids`: token ID序列
   - `attention_mask`: 注意力掩码，标识有效token
   - `labels`: 训练目标，通常是输出序列的token ID

## 3. 前向传播实现

SFT的前向传播主要在`CustomSeq2SeqTrainer`类中实现，该类继承自Hugging Face的`Seq2SeqTrainer`。

### 3.1 训练步骤

在训练过程中，前向传播的主要步骤如下：

1. **数据批次处理**：将输入数据转换为模型可接受的格式
2. **模型前向计算**：调用模型的forward方法计算logits和损失
3. **损失计算**：计算语言建模的交叉熵损失

### 3.2 前向传播代码流

在Trainer的训练循环中，每个步骤执行以下操作：

```python
# 在Trainer的training_step方法中
def training_step(self, model, inputs):
    model.train()
    inputs = self._prepare_inputs(inputs)
    
    with self.compute_loss_context_manager():
        loss = self.compute_loss(model, inputs)
        
    if self.args.gradient_accumulation_steps > 1:
        loss = loss / self.args.gradient_accumulation_steps
        
    loss.backward()
    
    return loss.detach()
```

`compute_loss`方法是执行实际前向传播的地方：

```python
def compute_loss(self, model, inputs, return_outputs=False):
    # 模型前向传播，获取输出
    outputs = model(**inputs)
    
    # 从输出中提取损失
    loss = outputs.loss
    
    return (loss, outputs) if return_outputs else loss
```

### 3.3 推理过程

在推理（评估或预测）阶段，SFT使用`prediction_step`方法来生成文本：

```python
def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
    if self.args.predict_with_generate:  # 生成模式
        labels = inputs.pop("labels", None)
    else:
        labels = inputs.get("labels")

    loss, generated_tokens, _ = super().prediction_step(
        model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
    )
    
    if generated_tokens is not None and self.args.predict_with_generate:
        # 移除prompt部分的生成结果
        generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
        generated_tokens = generated_tokens.contiguous()

    return loss, generated_tokens, labels
```

## 4. 损失函数计算

SFT使用标准的语言建模损失函数，即交叉熵损失。在计算过程中：

1. 模型输出logits，形状为`[batch_size, sequence_length, vocab_size]`
2. 与标签（通常是右移一个位置的输入）计算交叉熵
3. 通常会忽略pad token位置的损失

损失计算通常发生在模型的`forward`方法中：

```python
def forward(self, input_ids, attention_mask=None, labels=None, ...):
    # ... 其他处理 ...
    
    # 计算logits
    outputs = self.model(input_ids, attention_mask=attention_mask, ...)
    logits = outputs.logits
    
    loss = None
    if labels is not None:
        # 计算损失（忽略pad token）
        # 通常使用标准交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
    
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        # ... 其他输出 ...
    )
```

## 5. 与其他算法的区别

与其他高级微调算法相比，SFT有以下主要区别：

1. **与DPO/KTO的区别**：
   - SFT使用单一指令-响应对数据，而DPO/KTO使用偏好对（优选/拒绝响应）
   - SFT优化标准语言建模目标，而DPO/KTO优化基于偏好的目标

2. **与PPO的区别**：
   - SFT是简单的监督学习，而PPO是强化学习方法
   - SFT直接学习输入到输出的映射，而PPO通过奖励信号优化策略

3. **与PT的区别**：
   - PT专注于继续预训练任务，通常是无监督的
   - SFT专注于特定下游任务的映射学习，是有监督的

## 6. 总结

SFT是LLaMA Factory中最基础也是实现最简单的微调方法。它使用标准的seq2seq训练范式，将LLM作为条件文本生成模型训练，在给定提示(prompt)的条件下生成特定响应(response)。

其前向传播过程主要包括：
1. 输入编码与掩码生成
2. 模型前向计算得到logits
3. 计算交叉熵损失
4. 反向传播更新模型参数

SFT通常作为其他高级微调方法的第一阶段，为模型提供基础能力，然后再通过偏好学习或强化学习等方法进一步优化。 