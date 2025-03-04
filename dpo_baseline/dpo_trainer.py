#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DPO训练器实现，将训练过程拆解为forward、loss计算和backward三个阶段
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from dataclasses import dataclass


@dataclass
class DPOTrainerOutput:
    """
    用于存储DPO训练过程中的各个阶段输出
    """
    policy_chosen_logps: torch.FloatTensor = None
    policy_rejected_logps: torch.FloatTensor = None
    reference_chosen_logps: torch.FloatTensor = None
    reference_rejected_logps: torch.FloatTensor = None
    policy_chosen_logits: torch.FloatTensor = None
    policy_rejected_logits: torch.FloatTensor = None
    losses: torch.FloatTensor = None
    rewards: torch.FloatTensor = None
    

class DPOTrainer:
    """
    DPO训练器，拆解训练过程为forward、loss计算和backward三个阶段
    """
    
    def __init__(
        self,
        model_path: str,
        ref_model_path: str = None,
        beta: float = 0.1,
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        debug: bool = False,
    ):
        """
        初始化DPO训练器
        
        Args:
            model_path: 模型路径
            ref_model_path: 参考模型路径，如果为None则使用model_path的副本
            beta: DPO温度参数
            max_length: 最大序列长度
            device: 训练设备
            debug: 是否开启调试模式
        """
        self.beta = beta
        self.max_length = max_length
        self.device = device
        self.debug = debug
        
        # 加载策略模型（需要优化的模型）
        print(f"加载策略模型: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载参考模型（固定的SFT模型）
        if ref_model_path is None:
            print("未指定参考模型，复制策略模型作为参考模型")
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_path)
            for param in self.ref_model.parameters():
                param.requires_grad = False
        else:
            print(f"加载参考模型: {ref_model_path}")
            self.ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path)
        
        self.ref_model.to(device)
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def tokenize_batch(
        self, 
        prompts: List[str], 
        chosen_responses: List[str], 
        rejected_responses: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        对批次数据进行分词处理
        """
        # 构建输入格式: prompt + response
        chosen_texts = [p + c for p, c in zip(prompts, chosen_responses)]
        rejected_texts = [p + r for p, r in zip(prompts, rejected_responses)]
        
        # 分词
        chosen_tokens = self.tokenizer(chosen_texts, padding=True, truncation=True, 
                                        max_length=self.max_length, return_tensors="pt")
        rejected_tokens = self.tokenizer(rejected_texts, padding=True, truncation=True, 
                                         max_length=self.max_length, return_tensors="pt")
        
        # 分词处理提示部分，用于计算response的概率
        prompt_tokens = self.tokenizer(prompts, padding=True, truncation=True, 
                                       max_length=self.max_length, return_tensors="pt")
        
        # 构建response的mask
        chosen_response_mask = torch.ones_like(chosen_tokens["input_ids"]).bool()
        rejected_response_mask = torch.ones_like(rejected_tokens["input_ids"]).bool()
        
        for i, (chosen_text, rejected_text, prompt) in enumerate(zip(chosen_texts, rejected_texts, prompts)):
            prompt_len = len(self.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
            chosen_response_mask[i, :prompt_len] = False
            rejected_response_mask[i, :prompt_len] = False
        
        batch = {
            "chosen_input_ids": chosen_tokens["input_ids"].to(self.device),
            "chosen_attention_mask": chosen_tokens["attention_mask"].to(self.device),
            "rejected_input_ids": rejected_tokens["input_ids"].to(self.device),
            "rejected_attention_mask": rejected_tokens["attention_mask"].to(self.device),
            "chosen_response_mask": chosen_response_mask.to(self.device),
            "rejected_response_mask": rejected_response_mask.to(self.device),
        }
        
        return batch
    
    def forward(
        self, 
        batch: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Tuple[torch.FloatTensor, DPOTrainerOutput]:
        """
        前向传播阶段：计算策略模型和参考模型对偏好和非偏好回答的概率
        
        返回:
            loss: DPO损失
            outputs: 包含各种中间结果的对象
        """
        # 步骤1: 策略模型的前向传播
        policy_chosen_logits = self.model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"]
        ).logits
        
        policy_rejected_logits = self.model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"]
        ).logits
        
        # 步骤2: 参考模型的前向传播
        with torch.no_grad():
            reference_chosen_logits = self.ref_model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"]
            ).logits
            
            reference_rejected_logits = self.ref_model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"]
            ).logits
        
        # 步骤3: 计算log概率
        policy_chosen_logps = self._get_batch_logps(
            policy_chosen_logits, 
            batch["chosen_input_ids"], 
            batch["chosen_response_mask"]
        )
        
        policy_rejected_logps = self._get_batch_logps(
            policy_rejected_logits, 
            batch["rejected_input_ids"], 
            batch["rejected_response_mask"]
        )
        
        reference_chosen_logps = self._get_batch_logps(
            reference_chosen_logits, 
            batch["chosen_input_ids"], 
            batch["chosen_response_mask"]
        )
        
        reference_rejected_logps = self._get_batch_logps(
            reference_rejected_logits, 
            batch["rejected_input_ids"], 
            batch["rejected_response_mask"]
        )
        
        if self.debug:
            print(f"策略模型-偏好回答logp: {policy_chosen_logps.mean().item():.4f}")
            print(f"策略模型-拒绝回答logp: {policy_rejected_logps.mean().item():.4f}")
            print(f"参考模型-偏好回答logp: {reference_chosen_logps.mean().item():.4f}")
            print(f"参考模型-拒绝回答logp: {reference_rejected_logps.mean().item():.4f}")
        
        # 构建返回结果
        dpo_outputs = DPOTrainerOutput(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            policy_chosen_logits=policy_chosen_logits,
            policy_rejected_logits=policy_rejected_logits,
        )
        
        # 计算损失
        loss = self.compute_loss(dpo_outputs)
        dpo_outputs.losses = loss
        
        return loss, dpo_outputs
    
    def compute_loss(self, outputs: DPOTrainerOutput) -> torch.FloatTensor:
        """
        计算DPO损失
        
        Args:
            outputs: 前向传播的输出结果
            
        返回:
            loss: DPO损失
        """
        # 计算策略模型和参考模型的log概率比值
        policy_chosen_logps = outputs.policy_chosen_logps
        policy_rejected_logps = outputs.policy_rejected_logps
        reference_chosen_logps = outputs.reference_chosen_logps
        reference_rejected_logps = outputs.reference_rejected_logps
        
        # 计算奖励（隐式奖励）
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        
        # 计算DPO的logits (logit = r_chosen - r_rejected)
        reward_logits = self.beta * (
            (policy_chosen_logps - reference_chosen_logps) - 
            (policy_rejected_logps - reference_rejected_logps)
        )
        
        # 使用交叉熵损失进行优化（目标是将偏好和非偏好回答正确分类）
        losses = -F.logsigmoid(reward_logits)
        
        # 保存奖励信息用于分析
        outputs.rewards = torch.cat([chosen_rewards, rejected_rewards], dim=0)
        
        if self.debug:
            print(f"奖励差值: {reward_logits.mean().item():.4f}")
            print(f"平均损失: {losses.mean().item():.4f}")
            
        return losses.mean()
    
    def backward(self, loss: torch.FloatTensor) -> None:
        """
        反向传播阶段
        
        Args:
            loss: DPO损失
        """
        loss.backward()
    
    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        input_ids: torch.LongTensor,
        response_mask: torch.BoolTensor
    ) -> torch.FloatTensor:
        """
        计算每个样本的对数概率
        
        Args:
            logits: 模型logits输出，形状为[batch_size, seq_len, vocab_size]
            input_ids: 输入token ids，形状为[batch_size, seq_len]
            response_mask: 回答部分的mask，形状为[batch_size, seq_len]
            
        返回:
            batch_logps: 每个样本的对数概率，形状为[batch_size]
        """
        # 前移logits以便计算下一个token的概率
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        response_mask = response_mask[:, 1:]
        
        # 计算log概率
        log_probs = F.log_softmax(logits, dim=-1)
        token_logps = torch.gather(log_probs, -1, input_ids.unsqueeze(-1)).squeeze(-1)
        
        # 只计算response部分的log概率
        token_logps = token_logps * response_mask
        
        # 对每个样本求和，得到序列的log概率
        batch_logps = token_logps.sum(dim=-1) / response_mask.sum(dim=-1)
        
        return batch_logps
    
    def train_step(
        self,
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: List[str],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        执行一个训练步骤
        
        Args:
            prompts: 提示列表
            chosen_responses: 偏好回答列表
            rejected_responses: 非偏好回答列表
            optimizer: 优化器
            
        返回:
            metrics: 包含损失和其他指标的字典
        """
        # 准备批次数据
        batch = self.tokenize_batch(prompts, chosen_responses, rejected_responses)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        loss, outputs = self.forward(batch)
        
        # 打印中间结果
        if self.debug:
            chosen_reward = outputs.rewards[:len(prompts)].mean().item()
            rejected_reward = outputs.rewards[len(prompts):].mean().item()
            print(f"偏好回答平均奖励: {chosen_reward:.4f}")
            print(f"非偏好回答平均奖励: {rejected_reward:.4f}")
            print(f"奖励差值: {chosen_reward - rejected_reward:.4f}")
        
        # 反向传播
        self.backward(loss)
        
        # 更新参数
        optimizer.step()
        
        # 收集指标
        metrics = {
            "loss": loss.item(),
            "policy_chosen_logp": outputs.policy_chosen_logps.mean().item(),
            "policy_rejected_logp": outputs.policy_rejected_logps.mean().item(),
            "reference_chosen_logp": outputs.reference_chosen_logps.mean().item(),
            "reference_rejected_logp": outputs.reference_rejected_logps.mean().item(),
        }
        
        return metrics
    
    def save_model(self, output_dir: str) -> None:
        """保存模型"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"模型已保存到: {output_dir}")


if __name__ == "__main__":
    # 简单测试代码
    trainer = DPOTrainer(
        model_path="/root/models/Qwen1.5-0.5B",  # 使用本地模型
        beta=0.1,
        debug=True
    )
    
    # 示例数据
    prompts = ["请推荐一些健康的食品。"]
    chosen_responses = ["水果、蔬菜、全谷物和瘦肉是健康饮食的基础。特别推荐蓝莓、菠菜、三文鱼和燕麦，它们富含抗氧化剂和必需营养素。"]
    rejected_responses = ["汉堡、薯条和可乐是不错的选择，美味又方便。"]
    
    # 创建优化器
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=5e-5)
    
    # 执行训练步骤
    metrics = trainer.train_step(prompts, chosen_responses, rejected_responses, optimizer)
    
    print("训练指标:", metrics) 