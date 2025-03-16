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
        # 初始化beta_scale为10.0，这是一个全局缩放因子，使用更大的初始值
        self.beta_scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        
        # 构建更简单的value head网络，减少层数，避免梯度消失
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),  # 使用GELU激活函数代替ReLU
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()  # 确保输出为正值
        )   
        
        # 初始化网络权重
        self.value_head.apply(self.init_weights)
        
        # 设置beta的最小值和最大值
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # 打印初始化信息
        print(f"[INFO] ValueHead initialized with beta_scale={self.beta_scale.item():.4f}, beta_min={beta_min}, beta_max={beta_max}")
        

    def init_weights(self, module):
        """初始化网络权重，使用更大的标准差"""
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为0.1（比原来的0.01大）
            nn.init.normal_(module.weight, mean=0.0, std=0.1)
            # 将偏置初始化为小正数，而不是0，以避免初始输出过小
            nn.init.constant_(module.bias, 0.1)
            
            # 打印权重初始化信息
            print(f"[DEBUG] Initialized Linear layer: weight_shape={module.weight.shape}, bias_shape={module.bias.shape}")
            print(f"[DEBUG] Weight stats: mean={module.weight.mean().item():.4f}, std={module.weight.std().item():.4f}")
            print(f"[DEBUG] Bias stats: mean={module.bias.mean().item():.4f}, std={module.bias.std().item():.4f}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        Args:
            hidden_states: 最后一层hidden states，形状为[batch_size, hidden_size]
            
        Returns:
            beta值，形状为[batch_size]
        """
        # 输入是最后一层hidden states
        raw_beta = self.value_head(hidden_states)
        
        # 应用全局缩放因子
        scaled_beta = self.beta_scale * raw_beta
        
        # 将输出截断到[beta_min, beta_max]范围
        clamped_beta = torch.clamp(scaled_beta, min=self.beta_min, max=self.beta_max)
        
        # 打印调试信息
        if torch.rand(1).item() < 0.05:  # 增加打印概率到5%
            print(f"[DEBUG] ValueHead forward - raw_beta: min={raw_beta.min().item():.4f}, max={raw_beta.max().item():.4f}, mean={raw_beta.mean().item():.4f}")
            print(f"[DEBUG] ValueHead forward - beta_scale: {self.beta_scale.item():.4f}")
            print(f"[DEBUG] ValueHead forward - scaled_beta: min={scaled_beta.min().item():.4f}, max={scaled_beta.max().item():.4f}, mean={scaled_beta.mean().item():.4f}")
            print(f"[DEBUG] ValueHead forward - clamped_beta: min={clamped_beta.min().item():.4f}, max={clamped_beta.max().item():.4f}, mean={clamped_beta.mean().item():.4f}")
        
        return clamped_beta


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
        self.use_dynamic_beta = finetuning_args.use_dynamic_beta if hasattr(finetuning_args, "use_dynamic_beta") else False
        self.beta_min = finetuning_args.beta_min if hasattr(finetuning_args, "beta_min") else 0.01
        self.beta_max = finetuning_args.beta_max if hasattr(finetuning_args, "beta_max") else 1000.0

        # 添加对freeze_policy_model的支持
        self.freeze_policy_model = finetuning_args.freeze_policy_model if hasattr(finetuning_args, "freeze_policy_model") else False






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
            import os
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
        
        # 添加调试信息
        print(f"[DEBUG] DPO Loss - Delta stats: min={delta.min().item():.4f}, max={delta.max().item():.4f}, mean={delta.mean().item():.4f}")
        print(f"[DEBUG] DPO Loss - Logits after beta: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
        
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
        
        # 添加调试信息
        print(f"[DEBUG] DPO Loss - Loss stats: min={losses.min().item():.4f}, max={losses.max().item():.4f}, mean={losses.mean().item():.4f}")
        print(f"[DEBUG] DPO Loss - Chosen rewards: min={chosen_rewards.min().item():.4f}, max={chosen_rewards.max().item():.4f}, mean={chosen_rewards.mean().item():.4f}")
        print(f"[DEBUG] DPO Loss - Rejected rewards: min={rejected_rewards.min().item():.4f}, max={rejected_rewards.max().item():.4f}, mean={rejected_rewards.mean().item():.4f}")
  
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
        """
        使用动态beta计算preference loss
        """
        # 初始化dynamic_beta和beta
        dynamic_beta = self.beta  # 默认使用固定beta
        
        # 如果使用动态beta且提供了hidden states
        if self.use_dynamic_beta and chosen_prompt_last_token_hidden is not None:
            # 计算dynamic_beta
            dynamic_beta = self.value_head(chosen_prompt_last_token_hidden).squeeze(-1)
            
            # 添加调试信息
            print(f"[DEBUG] ValueHead beta_scale: {self.value_head.beta_scale.item():.4f}")
            print(f"[DEBUG] Dynamic beta stats: min={dynamic_beta.min().item():.4f}, max={dynamic_beta.max().item():.4f}, mean={dynamic_beta.mean().item():.4f}")
            
        # 计算delta值(如果使用参考模型)
        if reference_chosen_logps is not None and reference_rejected_logps is not None:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps
            delta = pi_logratios - ref_logratios
            
            # 添加调试信息
            positive_delta_mask = (delta > 0)
            negative_delta_mask = (delta < 0)
            pos_count = positive_delta_mask.sum().item()
            neg_count = negative_delta_mask.sum().item()
            
            print(f"[DEBUG] Delta stats: min={delta.min().item():.4f}, max={delta.max().item():.4f}, mean={delta.mean().item():.4f}")
            print(f"[DEBUG] Delta distribution: positive={pos_count}, negative={neg_count}, ratio={pos_count/(pos_count+neg_count):.4f}")
            
            if self.use_dynamic_beta:
                # 计算正负delta样本对应的beta平均值
                positive_beta_avg = dynamic_beta[positive_delta_mask].mean().item() if pos_count > 0 else 0
                negative_beta_avg = dynamic_beta[negative_delta_mask].mean().item() if neg_count > 0 else 0
                print(f"[DEBUG] Beta for positive delta: {positive_beta_avg:.4f}, Beta for negative delta: {negative_beta_avg:.4f}")
            
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
            # 根据use_disco参数决定使用哪种损失计算方法
            if self.use_disco:
                # 使用DISCO损失计算方法
                losses, chosen_rewards, rejected_rewards = self.disco_loss(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, dynamic_beta
                )
            else:
                # 使用标准DPO损失计算方法
                losses, chosen_rewards, rejected_rewards = self.dpo_loss_with_dynamic_beta(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, dynamic_beta
                )

        return losses, chosen_rewards, rejected_rewards, dynamic_beta

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        """
        计算DPO loss和其他指标，包括动态beta值
        """
        metrics = {}

        if self.finetuning_args.freeze_policy_model:
            # 冻结策略梯度更新
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
            with torch.no_grad(), ref_context:
                # 前向传播获取logps和hidden states
                outputs = self.concatenated_forward(model, batch)
        else:
            outputs = self.concatenated_forward(model, batch)
        
        # 解包输出，处理hidden states
        if len(outputs) == 6:  # 如果返回了hidden states
            policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits, policy_chosen_logps_avg, chosen_prompt_last_token_hidden = outputs
        else:  # 兼容旧版本
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
        
        # 记录delta和beta统计信息 - 新增代码
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
            
            # 记录指标
            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}delta/mean"] = delta.mean().item()
            metrics[f"{prefix}beta/positive_delta_avg"] = positive_beta_avg
            metrics[f"{prefix}beta/negative_delta_avg"] = negative_beta_avg
            metrics[f"{prefix}delta/positive_count"] = pos_count
            metrics[f"{prefix}delta/negative_count"] = neg_count
            metrics[f"{prefix}delta/positive_ratio"] = pos_count / (pos_count + neg_count)
        
        # 计算SFT损失并添加到总损失
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses = losses + self.ftx_gamma * sft_loss

        # 记录指标
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
        
        # 记录动态beta相关指标
        if self.use_dynamic_beta:
            metrics[f"{prefix}beta/value"] = dynamic_beta.mean().item()
            # metrics[f"{prefix}beta/min"] = dynamic_beta.min().item() 
            # metrics[f"{prefix}beta/max"] = dynamic_beta.max().item()
            metrics[f"{prefix}beta/scale"] = self.value_head.beta_scale.item()  
        
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
        
        # 记录DISCO特定指标
        if self.use_disco:
            # metrics[f"{prefix}disco/active"] = 1.0
            # metrics[f"{prefix}disco/variance"] = self.disco_variance
            if reference_chosen_logps is not None and reference_rejected_logps is not None:
                # 计算标准DPO和DISCO方法的偏好概率进行对比
                disco_probs = self.compute_preference_probability(
                    policy_chosen_logps, 
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                    method="disco"
                )
                standard_probs = self.compute_preference_probability(
                    policy_chosen_logps, 
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                    method="standard"
                )
                
                # # 记录偏好概率的平均值
                # metrics[f"{prefix}disco/pref_prob"] = disco_probs.mean().item()
                # metrics[f"{prefix}disco/std_pref_prob"] = standard_probs.mean().item()
                # metrics[f"{prefix}disco/pref_prob_diff"] = (disco_probs - standard_probs).mean().item()
                
                # 记录概率值的分布信息
                metrics[f"{prefix}disco/pref_prob_min"] = disco_probs.min().item()
                metrics[f"{prefix}disco/pref_prob_max"] = disco_probs.max().item()
                
                # 记录模型是否正确预测偏好的比例
                correct_pref = (disco_probs > 0.5).float().mean().item()
                metrics[f"{prefix}disco/correct_preference"] = correct_pref
   
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
        """
        重写父类方法以接受model参数
        """
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps

