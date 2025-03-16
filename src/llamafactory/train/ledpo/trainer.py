# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union, Any
import os
import math
from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps, nested_detach


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class ValueHead(nn.Module):
    """
    简单的value head网络，用于预测每个样本的beta值
    """
    
    def __init__(self, hidden_size: int, beta_min: float = 0.01, beta_max: float = 100.0):
        super().__init__()
        # 初始化beta_scale为10.0
        self.beta_scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
        # 构建value head网络
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()  # 将输出映射到0-1之间
        )   
        
        # 初始化网络权重
        self._init_weights()
        
        # 设置beta的最小值和最大值
        self.beta_min = beta_min
        self.beta_max = beta_max
        
    def _init_weights(self):
        """初始化网络权重"""
        for name, module in self.value_head.named_modules():
            if isinstance(module, nn.Linear):
                if name == '2':  # 最后一层
                    nn.init.normal_(module.weight, mean=0.1, std=0.2)
                    nn.init.constant_(module.bias, 0.5)  # 正偏置使得初始输出更大
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=0.1)
                    nn.init.constant_(module.bias, 0.1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播函数"""
        # 获取0-1范围的原始beta值
        raw_beta = self.value_head(hidden_states)
        
        # 确保beta_scale始终为正
        beta_scale_positive = F.softplus(self.beta_scale)
        
        # 将0-1映射到beta_min到beta_max范围
        beta_range = self.beta_max - self.beta_min
        scaled_beta = self.beta_min + raw_beta.squeeze(-1) * beta_range * (beta_scale_positive / 10.0)
        
        return scaled_beta


class LEDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        # 添加对value head的支持
        self.use_dynamic_beta = finetuning_args.use_dynamic_beta if hasattr(finetuning_args, "use_dynamic_beta") else True
        self.beta_min = finetuning_args.beta_min if hasattr(finetuning_args, "beta_min") else 0.001
        self.beta_max = finetuning_args.beta_max if hasattr(finetuning_args, "beta_max") else 1000.0

        # 添加对freeze_policy_model的支持
        self.freeze_policy_model = finetuning_args.freeze_policy_model if hasattr(finetuning_args, "freeze_policy_model") else False

        # 添加beta分析记录
        self.beta_history = {
            "train": {
                "steps": [],
                "beta_scale": [],
                "pos_beta": [],
                "neg_beta": [],
                "pos_neg_ratio": [],
                "loss": []
            },
            "eval": {
                "steps": [],
                "beta_scale": [],
                "pos_beta": [],
                "neg_beta": [],
                "pos_neg_ratio": [],
                "loss": []
            }
        }
        self.global_step = 0
        self.plot_dir = os.path.join(kwargs.get("args").output_dir, "beta_analysis")
        os.makedirs(self.plot_dir, exist_ok=True)

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma
        # 读取disco参数
        self.use_disco = finetuning_args.use_disco if hasattr(finetuning_args, "use_disco") else False
        self.disco_variance = finetuning_args.disco_variance if hasattr(finetuning_args, "disco_variance") else 1.0
        
        # 创建value head（只有在use_dynamic_beta为True时才会使用）
        if self.use_dynamic_beta:
            hidden_size = model.config.hidden_size
            self.value_head = ValueHead(hidden_size, self.beta_min, self.beta_max)
            # 将value_head放到与模型相同的设备上 - 尝试提前放置
            if hasattr(model, "device"):
                self.value_head = self.value_head.to(model.device)

        Trainer.__init__(self, model=model, **kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)


        # 设置 wandb_project (如果提供)
        append_str = "_disco" if self.use_disco else ""
        append_str += "_dynamic_beta" if self.use_dynamic_beta else ""
        append_str += "_freeze_policy_model" if self.freeze_policy_model else ""

        if hasattr(finetuning_args, 'wandb_project') and finetuning_args.wandb_project:
            os.environ["WANDB_PROJECT"] = finetuning_args.wandb_project 




    def __post_init__(self):
        """初始化后的设置"""
        super().__post_init__()
        
        # 确保value_head参数可训练
        if self.use_dynamic_beta and hasattr(self, "value_head"):
            for param in self.value_head.parameters():
                param.requires_grad = True

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            # 创建基本优化器
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)

            if self.optimizer is None:
                self.optimizer = super().create_optimizer()
                
            # 添加value_head参数到优化器
            if self.use_dynamic_beta and hasattr(self, "value_head"):
                # 确保value_head在正确设备上
                if hasattr(self.model, "device"):
                    self.value_head = self.value_head.to(self.model.device)
                
                # 获取value_head参数
                value_head_params = list(self.value_head.parameters())
                
                # 打印value_head参数信息
                print(f"[DEBUG] ValueHead parameters:")
                for name, param in self.value_head.named_parameters():
                    print(f"[DEBUG]   {name}: shape={param.shape}, requires_grad={param.requires_grad}")
                
                # 为value_head参数设置更高的学习率（例如，是基本学习率的10倍）
                value_head_lr = self.args.learning_rate * 10.0
                
                # 添加参数组
                params_config = {
                    "params": value_head_params,
                    "lr": value_head_lr,  # 使用更高的学习率
                }
                
                # 复制原优化器配置（除了学习率和参数）
                for k, v in self.optimizer.param_groups[0].items():
                    if k != "params" and k != "lr":
                        params_config[k] = v
                
                # 添加参数组
                self.optimizer.add_param_group(params_config)
                
                # 打印优化器信息
                print(f"[DEBUG] Optimizer param groups:")
                for i, group in enumerate(self.optimizer.param_groups):
                    print(f"[DEBUG]   Group {i}: {len(group['params'])} parameters, lr={group['lr']}")
                
                print(f"[DEBUG] ValueHead parameters added to optimizer with lr={value_head_lr}")
                print(f"[DEBUG] beta_scale initial value: {self.value_head.beta_scale.item():.4f}")

        return self.optimizer

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def get_batch_samples(self, epoch_iterator, num_batches):
        r"""
        Replaces the method of KTO Trainer with the one of the standard Trainer.
        """
        return Trainer.get_batch_samples(self, epoch_iterator, num_batches)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor", use_disco: bool = False) -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss
    
    def disco_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: "torch.Tensor",
        reference_rejected_logps: "torch.Tensor",
        dynamic_beta: "torch.Tensor" = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        实现DISCO (Difference In Squared Cumulative Optimization)损失
        
        DISCO使用标准正态分布的CDF来计算偏好概率，而不是标准DPO中的sigmoid函数。
        公式：P(a1 > a2 | s) = Φ((μ(a1) - μ(a2)) / sqrt(σ²(a1) + σ²(a2)))
        """
        # 使用Disco方法计算偏好概率
        preference_probs = self.compute_preference_probability(
            policy_chosen_logps, 
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            method="disco"
        )
        
        # 为了数值稳定性，对概率值进行裁剪，避免log(0)或log(1)的情况
        eps = 1e-6
        preference_probs = torch.clamp(preference_probs, min=eps, max=1-eps)
        
        # 计算损失：最大化偏好概率的对数似然
        losses = -torch.log(preference_probs)
        
        # 应用beta权重
        if dynamic_beta is not None:
            losses = dynamic_beta * losses
        else:
            losses = self.beta * losses
        
        # 计算chosen和rejected样本的奖励
        chosen_rewards = policy_chosen_logps - reference_chosen_logps
        rejected_rewards = policy_rejected_logps - reference_rejected_logps
        
        return losses, chosen_rewards, rejected_rewards
    
    def compute_preference_probability(
        self, 
        chosen_logps: "torch.Tensor", 
        rejected_logps: "torch.Tensor", 
        reference_chosen_logps: Optional["torch.Tensor"] = None,
        reference_rejected_logps: Optional["torch.Tensor"] = None,
        method: Literal["standard", "disco"] = "standard",
    ) -> "torch.Tensor":
        """
        计算偏好概率 P(chosen > rejected | state)
        
        Args:
            chosen_logps: 策略模型对chosen样本的对数概率
            rejected_logps: 策略模型对rejected样本的对数概率
            reference_chosen_logps: 参考模型对chosen样本的对数概率，仅在使用参考模型时需要
            reference_rejected_logps: 参考模型对rejected样本的对数概率，仅在使用参考模型时需要
            method: 计算方法，"standard"使用标准DPO方法，"disco"使用Disco-DPO方法
            variance: 在Disco方法中使用的方差值，默认为1.0
            
        Returns:
            偏好概率张量
        """
        if method == "standard":
            # 标准DPO的偏好概率计算：使用对数概率差的sigmoid函数
            if reference_chosen_logps is not None and reference_rejected_logps is not None:
                # 使用参考模型的对数概率差
                pi_logratios = chosen_logps - rejected_logps
                ref_logratios = reference_chosen_logps - reference_rejected_logps
                logits = pi_logratios - ref_logratios
            else:
                # 不使用参考模型
                logits = chosen_logps - rejected_logps
                
            # 使用sigmoid函数计算偏好概率
            preference_probs = torch.sigmoid(logits)
            
        elif method == "disco":
            # Disco-DPO的偏好概率计算：使用标准正态分布的CDF
            if reference_chosen_logps is not None and reference_rejected_logps is not None:
                # 计算策略模型和参考模型的对数概率差
                chosen_diff = chosen_logps - reference_chosen_logps
                rejected_diff = rejected_logps - reference_rejected_logps
                
                # 使用公式: Φ((μ(chosen) - μ(rejected)) / sqrt(σ²(chosen) + σ²(rejected)))
                mu_diff = chosen_diff - rejected_diff
            else:
                # 不使用参考模型时，直接使用策略模型的对数概率差
                mu_diff = chosen_logps - rejected_logps
            
            # 计算标准化差异（标准差为sqrt(2*variance)）
            variance = self.disco_variance if hasattr(self, "disco_variance") else 1.0
            standardized_diff = mu_diff / torch.sqrt(torch.tensor(2.0 * variance, device=chosen_logps.device))
            
            # 使用标准正态分布的CDF: Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
            preference_probs = 0.5 * (1.0 + torch.erf(standardized_diff / torch.sqrt(torch.tensor(2.0, device=standardized_diff.device))))
        else:
            raise ValueError(f"Unknown preference probability method: {method}")
        
        return preference_probs
    
    def dpo_loss_with_dynamic_beta(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: "torch.Tensor",
        reference_rejected_logps: "torch.Tensor",
        dynamic_beta: "torch.Tensor" = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        实现支持动态beta的标准DPO损失
        """
        
        assert dynamic_beta is not None, "dynamic_beta is None"
        
        # 获取偏好概率的logits（log-odds）
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        delta = pi_logratios - ref_logratios  # 完整的 Delta 值
        
        # 应用动态beta - 这是关键部分
        logits = dynamic_beta * delta
        
        # 计算交叉熵损失
        if self.label_smoothing > 0:
            # 为标签应用平滑处理
            losses = (
                -self.label_smoothing * F.logsigmoid(-logits) - (1 - self.label_smoothing) * F.logsigmoid(logits)
            )
        else:
            losses = -F.logsigmoid(logits)
        
        # 计算奖励 - 也应用动态beta
        chosen_rewards = dynamic_beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = dynamic_beta * (policy_rejected_logps - reference_rejected_logps)
        
        return losses, chosen_rewards, rejected_rewards


    def get_prompt_lengths(self, batch: Dict[str, "torch.Tensor"]) -> torch.Tensor:
        """
        计算batch中每个样本的prompt长度
        
        在DPO/LEDPO训练中，batch包含成对的样本(chosen和rejected)
        标签中IGNORE_INDEX表示prompt部分
        
        返回: [batch_size//2] 形状的张量，只包含chosen样本的prompt长度
        """
        # DPO/LEDPO batch中结构: [chosen_1, ..., chosen_n, rejected_1, ..., rejected_n]
        # 为了正确找到prompt长度，我们只需要处理chosen部分
        
        # 获取batch总大小并计算单侧大小(chosen或rejected部分)
        total_batch_size = batch["input_ids"].size(0)
        batch_size = total_batch_size // 2  # 每侧的样本数
        
        # 只获取chosen部分
        chosen_labels = batch["labels"][:batch_size]  # [batch_size, seq_len]
        chosen_input_ids = batch["input_ids"][:batch_size]  # [batch_size, seq_len]
        
        # 创建prompt掩码: True表示prompt部分
        prompt_mask = (chosen_labels == IGNORE_INDEX)  # [batch_size, seq_len]
        
        # 排除padding位置 (padding token通常是0)
        valid_tokens_mask = (chosen_input_ids != self.padding_value) & prompt_mask
        
        # 计算每个序列中有效prompt token的数量
        prompt_lengths = valid_tokens_mask.sum(dim=1)
        
        # 确保长度至少为1 (避免边缘情况)
        prompt_lengths = torch.maximum(prompt_lengths, torch.ones_like(prompt_lengths))
        
        return prompt_lengths

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
        """
        扩展原方法，添加对hidden states的提取，用于value head
        """
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # avoid error

        # 获取模型输出，包括hidden states
        model_outputs = model(**batch, return_dict=True, use_cache=False, output_hidden_states=True) 
        all_logits = model_outputs.logits.to(torch.float32)
        
        # 提取最后一层的hidden states
        last_hidden_states = model_outputs.hidden_states[-1] if self.use_dynamic_beta else None
        
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        
        # 如果使用动态beta，分割hidden states并获取prompt的最后一个token的hidden state
        chosen_prompt_last_token_hidden = None
        if self.use_dynamic_beta and last_hidden_states is not None:
            chosen_hidden, rejected_hidden = last_hidden_states.split(batch_size, dim=0)
            
            # 使用我们的函数获取prompt长度
            prompt_lengths = self.get_prompt_lengths(batch)  # 这个函数已修改为只返回chosen部分
            
            if prompt_lengths.shape[0] > 0:  # 确保不是空batch
                # 创建批次索引 [0, 1, 2, ..., batch_size-1]
                batch_indices = torch.arange(batch_size, device=chosen_hidden.device)
                
                # 由于Python索引从0开始，将长度减1获取索引位置
                prompt_indices = (prompt_lengths - 1).clamp(0, chosen_hidden.size(1) - 1)
                
                # 使用批次索引和提示长度索引获取提示的最后一个token的hidden state
                chosen_prompt_last_token_hidden = chosen_hidden[batch_indices, prompt_indices]

        if self.loss_type in ["ipo", "orpo", "simpo"]:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps, chosen_prompt_last_token_hidden
        else:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length, chosen_prompt_last_token_hidden

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
        chosen_prompt_last_token_hidden: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
        """使用动态beta计算preference loss"""
        # 初始化dynamic_beta
        dynamic_beta = self.beta  # 默认使用固定beta
        
        # 计算delta值(如果使用参考模型)
        delta = None
        if reference_chosen_logps is not None and reference_rejected_logps is not None:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps
            delta = pi_logratios - ref_logratios
        
        # 如果使用动态beta且提供了hidden states
        if self.use_dynamic_beta and chosen_prompt_last_token_hidden is not None:
            # 计算dynamic_beta
            dynamic_beta = self.value_head(chosen_prompt_last_token_hidden)
            
            # 根据delta符号增强beta分化
            if delta is not None:
                # 创建delta>0和delta<0的掩码
                positive_delta_mask = (delta > 0)
                negative_delta_mask = (delta <= 0)
                
                # 记录原始beta值
                original_beta = dynamic_beta.clone()
                
                # 对于正delta，增加beta值；对于负delta，减小beta值
                beta_adjust_factor = 1.2  # 调整系数
                
                # 创建系数张量并应用调整
                beta_multiplier = torch.ones_like(dynamic_beta)
                beta_multiplier[positive_delta_mask] = beta_adjust_factor
                beta_multiplier[negative_delta_mask] = 1.0 / beta_adjust_factor
                
                # 应用调整
                dynamic_beta = dynamic_beta * beta_multiplier
        
        # 计算损失
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps, dynamic_beta)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps, dynamic_beta)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

            chosen_rewards = dynamic_beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = dynamic_beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            if self.use_disco:
                losses, chosen_rewards, rejected_rewards = self.disco_loss(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, dynamic_beta
                )
            else:
                losses, chosen_rewards, rejected_rewards = self.dpo_loss_with_dynamic_beta(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, dynamic_beta
                )
        
        # 添加beta_scale正则化项
        if self.use_dynamic_beta and hasattr(self, "value_head"):
            # 使用L2正则化防止beta_scale过小
            beta_scale_regularization = 0.01 * torch.square(10.0 - F.softplus(self.value_head.beta_scale))
            # 添加到损失中
            losses = losses + beta_scale_regularization
            
        return losses, chosen_rewards, rejected_rewards, dynamic_beta

    def beta_analysis(self, metrics: Dict[str, float], loss: float, train_eval: str = "train") -> None:
        """
        分析和记录beta相关指标
        
        Args:
            metrics: 训练/评估指标字典
            loss: 当前损失值
            train_eval: 'train'或'eval'，表示当前阶段
        """
        prefix = "eval_" if train_eval == "eval" else ""
        step = self.global_step
        
        # 提取beta指标
        beta_scale = metrics.get(f"{prefix}beta/scale", 0.0)
        pos_beta = metrics.get(f"{prefix}beta/positive_delta_avg", 0.0)
        neg_beta = metrics.get(f"{prefix}beta/negative_delta_avg", 0.0)
        pos_neg_ratio = metrics.get(f"{prefix}beta/pos_neg_ratio", 0.0)
        
        # 记录到历史数据中
        self.beta_history[train_eval]["steps"].append(step)
        self.beta_history[train_eval]["beta_scale"].append(beta_scale)
        self.beta_history[train_eval]["pos_beta"].append(pos_beta)
        self.beta_history[train_eval]["neg_beta"].append(neg_beta)
        self.beta_history[train_eval]["pos_neg_ratio"].append(pos_neg_ratio)
        self.beta_history[train_eval]["loss"].append(loss)
        
        # 每50步打印一次详细信息
        if train_eval == "train" and (step % 50 == 0 or step < 10):
            print(f"\n===== LEDPO Beta分析 Step {step} =====")
            print(f"beta_scale = {beta_scale:.4f}")
            print(f"正delta对应beta (pos_beta) = {pos_beta:.4f}")
            print(f"负delta对应beta (neg_beta) = {neg_beta:.4f}")
            print(f"pos_beta/neg_beta比值 = {pos_neg_ratio:.4f}")
            print(f"当前损失 = {loss:.4f}")
            print("="*35)
    
    def plot_beta_trends(self) -> None:
        """绘制beta相关指标的趋势图"""
        # 确保有足够的数据点
        if len(self.beta_history["train"]["steps"]) < 2:
            return
        
        # 创建绘图
        plt.figure(figsize=(20, 15))
        
        # 1. 绘制beta_scale变化
        plt.subplot(2, 2, 1)
        plt.plot(self.beta_history["train"]["steps"], self.beta_history["train"]["beta_scale"], 'b-', label='train')
        if self.beta_history["eval"]["steps"]:
            plt.plot(self.beta_history["eval"]["steps"], self.beta_history["eval"]["beta_scale"], 'r-', label='eval')
        plt.xlabel("Steps")
        plt.ylabel("Beta Scale")
        plt.title("Beta Scale Trend")
        plt.legend()
        plt.grid(True)
        
        # 2. 绘制pos_beta和neg_beta变化
        plt.subplot(2, 2, 2)
        plt.plot(self.beta_history["train"]["steps"], self.beta_history["train"]["pos_beta"], 'g-', label='train_pos_beta')
        plt.plot(self.beta_history["train"]["steps"], self.beta_history["train"]["neg_beta"], 'r-', label='train_neg_beta')
        if self.beta_history["eval"]["steps"]:
            plt.plot(self.beta_history["eval"]["steps"], self.beta_history["eval"]["pos_beta"], 'g--', label='eval_pos_beta')
            plt.plot(self.beta_history["eval"]["steps"], self.beta_history["eval"]["neg_beta"], 'r--', label='eval_neg_beta')
        plt.xlabel("Steps")
        plt.ylabel("Beta Value")
        plt.title("Positive/Negative Delta Beta Trends")
        plt.legend()
        plt.grid(True)
        
        # 3. 绘制pos_beta/neg_beta比值变化
        plt.subplot(2, 2, 3)
        plt.plot(self.beta_history["train"]["steps"], self.beta_history["train"]["pos_neg_ratio"], 'b-', label='train')
        if self.beta_history["eval"]["steps"]:
            plt.plot(self.beta_history["eval"]["steps"], self.beta_history["eval"]["pos_neg_ratio"], 'r-', label='eval')
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)  # 添加y=1的参考线
        plt.xlabel("Steps")
        plt.ylabel("Pos Beta / Neg Beta Ratio")
        plt.title("Ratio of Positive to Negative Delta Beta")
        plt.legend()
        plt.grid(True)
        
        # 4. 绘制损失变化
        plt.subplot(2, 2, 4)
        plt.plot(self.beta_history["train"]["steps"], self.beta_history["train"]["loss"], 'b-', label='train')
        if self.beta_history["eval"]["steps"]:
            plt.plot(self.beta_history["eval"]["steps"], self.beta_history["eval"]["loss"], 'r-', label='eval')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training/Evaluation Loss")
        plt.legend()
        plt.grid(True)
        
        plt.suptitle("LEDPO Beta Analysis", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图形
        plt.savefig(os.path.join(self.plot_dir, f"beta_trends_step_{self.global_step}.png"))
        plt.savefig(os.path.join(self.plot_dir, "beta_trends_latest.png"))  # 始终覆盖最新的
        plt.close()
        
        # 保存原始数据以便后续分析
        np.save(os.path.join(self.plot_dir, "beta_history.npy"), self.beta_history)
    
    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        """计算DPO loss和相关指标"""
        metrics = {}

        if self.finetuning_args.freeze_policy_model:
            # 冻结策略梯度更新 - 但不阻断ValueHead梯度
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
            with ref_context:
                # 前向传播获取logps和hidden states
                outputs = self.concatenated_forward(model, batch)
                
                # 分离策略模型的logps，但保留hidden states的梯度
                if len(outputs) == 6:
                    policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits, policy_chosen_logps_avg, chosen_prompt_last_token_hidden = outputs
                    # 只分离logps和logits，保持hidden states的梯度连接
                    policy_chosen_logps = policy_chosen_logps.detach()
                    policy_rejected_logps = policy_rejected_logps.detach()
                    policy_chosen_logits = policy_chosen_logits.detach()
                    policy_rejected_logits = policy_rejected_logits.detach()
                    policy_chosen_logps_avg = policy_chosen_logps_avg.detach()
                else:
                    policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits, policy_chosen_logps_avg = outputs
                    # 分离所有输出
                    policy_chosen_logps = policy_chosen_logps.detach()
                    policy_rejected_logps = policy_rejected_logps.detach()
                    policy_chosen_logits = policy_chosen_logits.detach()
                    policy_rejected_logits = policy_rejected_logits.detach()
                    policy_chosen_logps_avg = policy_chosen_logps_avg.detach()
                    chosen_prompt_last_token_hidden = None
        else:
            outputs = self.concatenated_forward(model, batch)
            if len(outputs) == 6:
                policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits, policy_chosen_logps_avg, chosen_prompt_last_token_hidden = outputs
            else:
                policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits, policy_chosen_logps_avg = outputs
                chosen_prompt_last_token_hidden = None

        # 获取参考模型的log probs
        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        
        # 计算损失和奖励
        losses, chosen_rewards, rejected_rewards, dynamic_beta = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            chosen_prompt_last_token_hidden,
        )
        
        # 记录delta和beta统计信息
        if self.use_dynamic_beta and reference_chosen_logps is not None and reference_rejected_logps is not None:
            # 计算delta值
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps
            delta = pi_logratios - ref_logratios
            
            # 创建delta>0和delta<0的掩码
            positive_delta_mask = (delta > 0).float()
            negative_delta_mask = (delta < 0).float()
            
            # 计算正负delta样本的数量
            pos_count = positive_delta_mask.sum().item()
            neg_count = negative_delta_mask.sum().item()
            
            # 防止除零错误
            pos_count = max(pos_count, 1.0)
            neg_count = max(neg_count, 1.0)
            
            # 计算正负delta样本对应的beta平均值
            positive_beta_avg = (dynamic_beta * positive_delta_mask).sum().item() / pos_count
            negative_beta_avg = (dynamic_beta * negative_delta_mask).sum().item() / neg_count
            pos_neg_ratio = positive_beta_avg / negative_beta_avg if negative_beta_avg > 0 else 0.0
            
            # 记录指标
            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}delta/mean"] = delta.mean().item()
            metrics[f"{prefix}beta/positive_delta_avg"] = positive_beta_avg
            metrics[f"{prefix}beta/negative_delta_avg"] = negative_beta_avg
            metrics[f"{prefix}beta/scale"] = F.softplus(self.value_head.beta_scale).item()
            metrics[f"{prefix}beta/pos_neg_ratio"] = pos_neg_ratio
        
        # 计算SFT损失并添加到总损失
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses = losses + self.ftx_gamma * sft_loss

        # 记录基本指标
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
        
        # 更新步数计数器(仅在训练阶段)
        if train_eval == "train":
            self.global_step += 1
            
            # 每隔一定步数绘制图表
            if self.global_step % 100 == 0:
                self.plot_beta_trends()
        
        # 分析beta指标
        self.beta_analysis(metrics, losses.mean().item(), train_eval)
        
        return losses.mean(), metrics

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        r"""
        Subclass and override to accept extra kwargs.
        """
        return super().compute_loss(model, inputs, return_outputs)

    @override
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        r"""
        Log `logs` on the various objects watching training, including stored metrics.
        """
        # logs either has "loss" or "eval_loss"
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 10:  # pad to for all reduce
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        for key, metric in zip(key_list, metric_list):  # add remaining items
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs, *args, **kwargs)
        
    # 添加重写compute_reference_log_probs方法
    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """重写父类方法以接受model参数"""
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        # 不使用torch.no_grad()，而是在获取输出后detach
        with ref_context:
            reference_outputs = self.concatenated_forward(ref_model, batch)
            
            # 根据输出长度解包并detach
            if len(reference_outputs) >= 2:
                reference_chosen_logps, reference_rejected_logps = reference_outputs[:2]
                # 分离梯度
                reference_chosen_logps = reference_chosen_logps.detach()
                reference_rejected_logps = reference_rejected_logps.detach()
            else:
                return None, None

        return reference_chosen_logps, reference_rejected_logps

    def analyze_beta_results(self, save_summary: bool = True) -> Dict[str, Any]:
        """
        分析beta训练结果并生成总结报告
        
        Args:
            save_summary: 是否保存总结到文件
            
        Returns:
            包含分析结果的字典
        """
        print("\n" + "="*50)
        print("LEDPO Beta 训练结果分析")
        print("="*50)
        
        # 检查是否有足够的数据
        if len(self.beta_history["train"]["steps"]) < 10:
            print("训练数据不足，无法进行完整分析")
            return {"success": False, "reason": "训练数据不足"}
        
        # 1. 提取关键指标
        train_data = self.beta_history["train"]
        
        # 获取最初和最后的beta_scale
        initial_beta_scale = train_data["beta_scale"][0] if train_data["beta_scale"] else 0.0
        final_beta_scale = train_data["beta_scale"][-1] if train_data["beta_scale"] else 0.0
        beta_scale_change = final_beta_scale - initial_beta_scale
        
        # 计算pos_beta和neg_beta的平均值（最后20%的步骤）
        last_20_percent = max(1, len(train_data["steps"]) // 5)
        final_pos_beta = np.mean(train_data["pos_beta"][-last_20_percent:])
        final_neg_beta = np.mean(train_data["neg_beta"][-last_20_percent:])
        final_pos_neg_ratio = final_pos_beta / final_neg_beta if final_neg_beta > 0 else 0.0
        
        # 2. 计算趋势
        # 检查beta_scale是否在下降
        beta_scale_trend = "下降" if beta_scale_change < -0.1 else "上升" if beta_scale_change > 0.1 else "稳定"
        
        # 检查beta值有没有趋零问题
        has_zero_beta_issue = final_pos_beta < self.beta_min * 2 or final_neg_beta < self.beta_min * 2
        
        # 检查pos_beta和neg_beta的分化情况
        has_good_differentiation = final_pos_neg_ratio > 1.2  # 理想情况下pos_beta应该明显大于neg_beta
        
        # 3. 总结结果
        result = {
            "success": not has_zero_beta_issue and has_good_differentiation,
            "initial_beta_scale": initial_beta_scale,
            "final_beta_scale": final_beta_scale,
            "beta_scale_change": beta_scale_change,
            "beta_scale_trend": beta_scale_trend,
            "final_pos_beta": final_pos_beta,
            "final_neg_beta": final_neg_beta,
            "final_pos_neg_ratio": final_pos_neg_ratio,
            "has_zero_beta_issue": has_zero_beta_issue,
            "has_good_differentiation": has_good_differentiation,
            "freeze_policy_model": self.freeze_policy_model,
        }
        
        # 4. 打印总结
        print(f"初始 beta_scale: {initial_beta_scale:.4f}")
        print(f"最终 beta_scale: {final_beta_scale:.4f} ({beta_scale_trend})")
        print(f"最终平均正delta beta值: {final_pos_beta:.4f}")
        print(f"最终平均负delta beta值: {final_neg_beta:.4f}")
        print(f"最终 pos_beta/neg_beta 比值: {final_pos_neg_ratio:.4f}")
        print(f"beta趋零问题: {'存在' if has_zero_beta_issue else '不存在'}")
        print(f"beta值分化: {'良好' if has_good_differentiation else '不足'}")
        print(f"freeze_policy_model设置: {self.freeze_policy_model}")
        
        # 结论
        print("\n结论:")
        if not has_zero_beta_issue and has_good_differentiation:
            print("✅ LEDPO训练正常，beta值表现良好。")
        else:
            print("❌ LEDPO训练异常，beta值可能存在问题。")
            
            if has_zero_beta_issue:
                print("  - beta值过小，可能出现了趋零问题")
                print("  - 建议: 关闭freeze_policy_model或增加beta_min")
                
            if not has_good_differentiation:
                print("  - pos_beta和neg_beta没有良好分化")
                print("  - 建议: 检查损失函数计算或增加正反例对比度")
        
        # 5. 保存总结到文件
        if save_summary:
            summary_path = os.path.join(self.plot_dir, "beta_analysis_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("LEDPO Beta 训练结果分析\n")
                f.write("="*30 + "\n\n")
                f.write(f"初始 beta_scale: {initial_beta_scale:.4f}\n")
                f.write(f"最终 beta_scale: {final_beta_scale:.4f} ({beta_scale_trend})\n")
                f.write(f"最终平均正delta beta值: {final_pos_beta:.4f}\n")
                f.write(f"最终平均负delta beta值: {final_neg_beta:.4f}\n")
                f.write(f"最终 pos_beta/neg_beta 比值: {final_pos_neg_ratio:.4f}\n")
                f.write(f"beta趋零问题: {'存在' if has_zero_beta_issue else '不存在'}\n")
                f.write(f"beta值分化: {'良好' if has_good_differentiation else '不足'}\n")
                f.write(f"freeze_policy_model设置: {self.freeze_policy_model}\n\n")
                
                f.write("结论:\n")
                if not has_zero_beta_issue and has_good_differentiation:
                    f.write("✅ LEDPO训练正常，beta值表现良好。\n")
                else:
                    f.write("❌ LEDPO训练异常，beta值可能存在问题。\n")
                    
                    if has_zero_beta_issue:
                        f.write("  - beta值过小，可能出现了趋零问题\n")
                        f.write("  - 建议: 关闭freeze_policy_model或增加beta_min\n")
                        
                    if not has_good_differentiation:
                        f.write("  - pos_beta和neg_beta没有良好分化\n")
                        f.write("  - 建议: 检查损失函数计算或增加正反例对比度\n")
                
            print(f"\n分析总结已保存到: {summary_path}")
        
        # 在分析完成后再次绘制最终图表
        self.plot_beta_trends()
        
        return result

    # 重写train方法以在训练结束后进行分析
    @override
    def train(self, *args, **kwargs):
        # 调用原始的train方法
        result = super().train(*args, **kwargs)
        
        # 训练结束后进行beta分析
        self.analyze_beta_results()
        
        return result

