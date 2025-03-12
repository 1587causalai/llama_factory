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
    
    def __init__(self, hidden_size: int, beta_min: float = 0.1, beta_max: float = 100.0):
        super().__init__()
        self.beta_scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1)
        )   
        self.value_head.apply(self.init_weights)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入是最后一层hidden states
        beta_raw = self.value_head(hidden_states)
        beta = nn.functional.softplus(beta_raw)
        # 使用将输出截断到[beta_min, beta_max]范围
        scale = torch.clamp(self.beta_scale , min=self.beta_min, max=self.beta_max)
        return beta * scale


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

        # 设置 wandb_project (如果提供)
        if hasattr(finetuning_args, 'wandb_project') and finetuning_args.wandb_project:
            import os
            os.environ["WANDB_PROJECT"] = finetuning_args.wandb_project

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma
        
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

    def __post_init__(self):
        """初始化后的设置"""
        super().__post_init__()
        
        # 确保value_head参数可训练
        if self.use_dynamic_beta and hasattr(self, "value_head") and hasattr(self.model, "set_adapter"):
            for param in self.value_head.parameters():
                param.requires_grad = True

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            # 创建基本优化器
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)

            if self.optimizer is None:
                self.optimizer = super().create_optimizer()
                
            # # 冻结策略模型的所有参数
            # print("正在冻结策略模型参数，只训练beta相关参数...")
            # for param in self.model.parameters():
            #     param.requires_grad = False
                    
            # 添加value_head参数到优化器
            if self.use_dynamic_beta and hasattr(self, "value_head"):
                # 确保value_head在正确设备上
                if hasattr(self.model, "device"):
                    self.value_head = self.value_head.to(self.model.device)
                
                # 添加value_head参数到优化器
                params_config = {"params": list(self.value_head.parameters())}
                
                # 复制原优化器配置
                for k, v in self.optimizer.param_groups[0].items():
                    if k != "params":
                        params_config[k] = v
                
                # 添加参数组
                self.optimizer.add_param_group(params_config)
                
                # 简单调试信息
                print(f"beta_scale已添加到优化器，初始值: {self.value_head.beta_scale.item()}")

        self.freeze_policy_model()
        
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

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
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
            
            pi_logratios = policy_chosen_logps - policy_rejected_logps  # shape = [batch_size]
            ref_logratios = reference_chosen_logps - reference_rejected_logps  # shape = [batch_size]
            
            logits = dynamic_beta * (pi_logratios - ref_logratios)  # shape = [batch_size], 防止 beta 收到负面样本影响, 总是想着减小beta
            
            if self.label_smoothing > 0:
                # 为标签应用平滑处理
                losses = (
                    -self.label_smoothing * F.logsigmoid(-logits) - (1 - self.label_smoothing) * F.logsigmoid(logits)
                )
            else:
                losses = -F.logsigmoid(logits)
            
            chosen_rewards = dynamic_beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_rewards = dynamic_beta * (policy_rejected_logps - reference_rejected_logps)

            # 计算完整的 Delta 值 (论文中的 Δ)
            delta = pi_logratios - ref_logratios  # 完整的 Delta 值
            
            # 打印详细信息
            print(f"=== 训练状态信息 ===")
        
            
            # 分析 Delta 与 beta 的关系
            for i in range(len(delta)):
                print(f"样本 {i}: Delta={delta[i]:.4f}, beta={dynamic_beta[i]:.4f} "
                      f"-> {'正向样本，期望beta增大' if delta[i] > 0 else '负向样本，期望beta减小'}")
            
            print(f"chosen_rewards: {chosen_rewards}")
            print(f"rejected_rewards: {rejected_rewards}")
            print("==================")
            
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
            # 调用支持动态beta的DPO损失计算函数
            # 这里的dynamic_beta会直接影响损失计算和奖励缩放
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

        # 冻结策略梯度更新
        ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        with torch.no_grad(), ref_context:
            # 前向传播获取logps和hidden states
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
            metrics[f"{prefix}beta/min"] = dynamic_beta.min().item()
            metrics[f"{prefix}beta/max"] = dynamic_beta.max().item()
            metrics[f"{prefix}beta/scale"] = self.value_head.beta_scale.item()  
        
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
        
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
        
    # 添加方法以冻结策略模型参数，只训练value head
    def freeze_policy_model(self):
        """冻结策略模型参数，只训练value head"""
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 确保value_head参数可训练
        if self.use_dynamic_beta:
            for param in self.value_head.parameters():
                param.requires_grad = True
                
            # 打印训练参数数量
            trainable_params = sum(p.numel() for p in self.value_head.parameters() if p.requires_grad)
            print(f"冻结策略模型后，可训练参数数量: {trainable_params}，仅包含value head网络参数")

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

