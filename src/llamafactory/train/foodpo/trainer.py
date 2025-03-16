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
from typing import TYPE_CHECKING, Literal, Optional, Union

from fastapi import logger
import torch
import torch.nn.functional as F
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


class CustomFooDPOTrainer(DPOTrainer):
    """
    FooDPO是一种基于DPO的新算法，用于增强模型对人类偏好的学习效果。
    这个训练器实现了FooDPO的核心功能。
    """
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

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        # dynamic beta
        self.use_dynamic_beta = getattr(finetuning_args, "use_dynamic_beta", False)
        self.freeze_policy = getattr(finetuning_args, "freeze_policy", False)
        
        if self.use_dynamic_beta:
            from .beta_head import HiddenStateBetaHead
            self.current_batch = None
            self.current_beta_values = None

            self.beta_head = HiddenStateBetaHead(
                hidden_size=model.config.hidden_size,
                beta_base=self.beta
            )
            

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
            
        if self.use_dynamic_beta:
            self.beta_head = self.beta_head.to(self.accelerator.device)
            
            # 根据freeze_policy参数决定是否冻结策略模型
            if self.freeze_policy:
                print("\n[FREEZE] 冻结策略模型参数，只训练beta_head...")
                for param in self.model.parameters():
                    param.requires_grad = False
                    
                # 确保beta_head参数可训练
                for param in self.beta_head.parameters():
                    param.requires_grad = True
                
                # 测试梯度流动
                self.test_grad_flow()
            
    def test_grad_flow(self):
        """测试beta_head梯度流动是否正常"""
        if not hasattr(self, "beta_head") or not self.use_dynamic_beta:
            return
            
        print("\n[GRAD-TEST] 开始测试beta_head梯度流...")
        
        # 1. 创建随机输入
        batch_size = 4
        hidden_size = self.model.config.hidden_size
        fake_hidden = torch.randn(batch_size, hidden_size, device=self.accelerator.device, requires_grad=True)
        
        # 2. 计算beta值
        beta_values = self.beta_head(fake_hidden)
        print(f"[GRAD-TEST] 计算的beta值: {beta_values}")
        
        # 3. 创建假损失并反向传播
        fake_loss = beta_values.mean()
        fake_loss.backward()
        
        # 4. 检查beta_head参数是否收到梯度
        has_grad = False
        for name, param in self.beta_head.named_parameters():
            if param.grad is not None and param.grad.norm() > 0:
                has_grad = True
                print(f"[GRAD-TEST] 参数 {name} 收到梯度: norm={param.grad.norm().item()}")
            else:
                print(f"[GRAD-TEST] 参数 {name} 没有收到梯度!")
        
        print(f"[GRAD-TEST] 梯度流动测试结果: {'成功' if has_grad else '失败'}")
        
        # 5. 清理梯度
        for param in self.beta_head.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def get_prompt_lengths(self, batch: dict[str, "torch.Tensor"]) -> torch.Tensor:
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
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            # 创建基本优化器
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)

            if self.optimizer is None:
                self.optimizer = super().create_optimizer()
            
            # 添加beta_head参数到优化器
            if self.use_dynamic_beta and hasattr(self, "beta_head"):
                # 确保beta_head在正确设备上
                if hasattr(self.model, "device"):
                    self.beta_head = self.beta_head.to(self.model.device)
                
                # 获取beta_head参数
                beta_head_params = list(self.beta_head.parameters())
                
                # 打印beta_head参数信息
                print(f"[DEBUG] BetaHead parameters:")
                for name, param in self.beta_head.named_parameters():
                    print(f"[DEBUG]   {name}: shape={param.shape}, requires_grad={param.requires_grad}")
                
                # 为beta_head参数设置更高的学习率（例如，是基本学习率的10倍）
                beta_head_lr = self.args.learning_rate * 10.0
                
                # 添加参数组
                params_config = {
                    "params": beta_head_params,
                    "lr": beta_head_lr,  # 使用更高的学习率
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
        r"""Replace the method of DPO Trainer with the one of the standard Trainer."""
        return Trainer.get_batch_samples(self, epoch_iterator, num_batches)

    def odds_ratio_loss(
        self, 
        chosen_logps: "torch.Tensor", 
        rejected_logps: "torch.Tensor",
        beta_values: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        r"""Compute ORPO's odds ratio (OR) loss for batched log probabilities of the policy model."""
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        
        # 使用动态 beta 或固定 beta
        beta = beta_values if beta_values is not None else self.beta
        orpo_loss = sft_loss + beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(
        self, 
        chosen_logps: "torch.Tensor", 
        rejected_logps: "torch.Tensor",
        beta_values: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        r"""Compute SimPO loss for batched log probabilities of the policy model."""
        pi_logratios = chosen_logps - rejected_logps
        
        # 使用动态 beta 或固定 beta
        beta = beta_values if beta_values is not None else self.beta
        gamma_logratios = self.simpo_gamma / beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(beta * logits)
        return simpo_loss

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""Compute the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # avoid error

        # 获取模型输出，包括logits和last_hidden_state, 如果use_dynamic_beta为True，则额外输出outputs_hidden_states
        if self.use_dynamic_beta:   
            # 正常获取模型输出，不要detach以保持梯度流
            model_outputs = model(**batch, return_dict=True, use_cache=False, output_hidden_states=True)
        else:
            model_outputs = model(**batch, return_dict=True, use_cache=False)

        all_logits: torch.Tensor = model_outputs.logits.to(torch.float32)
        
        # 获取last_hidden_state用于动态beta计算
        last_hidden_states = model_outputs.hidden_states[-1] if self.use_dynamic_beta else None
        
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        
        # 如果使用动态beta，计算beta值
        if self.use_dynamic_beta and last_hidden_states is not None:
            chosen_hidden, rejected_hidden = last_hidden_states.split(batch_size, dim=0)
            
            # 使用get_prompt_lengths获取prompt长度
            prompt_lengths = self.get_prompt_lengths(batch)  # 这个函数已修改为只返回chosen部分
            
            if prompt_lengths.shape[0] > 0:  # 确保不是空batch
                # 创建批次索引 [0, 1, 2, ..., batch_size-1]
                batch_indices = torch.arange(batch_size, device=chosen_hidden.device)
                
                # 由于Python索引从0开始，将长度减1获取索引位置
                prompt_indices = (prompt_lengths - 1).clamp(0, chosen_hidden.size(1) - 1)
                
                # 使用批次索引和提示长度索引获取提示的最后一个token的hidden state
                chosen_prompt_last_token_hidden = chosen_hidden[batch_indices, prompt_indices]
                
                # 使用beta_head计算动态beta值
                self.current_beta_values = self.beta_head(chosen_prompt_last_token_hidden)

        if self.loss_type in ["ipo", "orpo", "simpo"]:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps
        else:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
    ) -> tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Compute log probabilities of the reference model."""
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

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        r"""Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
        
        # 添加 delta 值指标 - 无论是否使用动态beta都记录
        if hasattr(self, "current_delta_values") and self.current_delta_values is not None:
            metrics[f"{prefix}delta"] = self.current_delta_values.mean().item()
        
        # 添加 beta 相关指标
        beta = self.current_beta_values if self.use_dynamic_beta and self.current_beta_values is not None else self.beta
        
        if self.use_dynamic_beta and hasattr(self, "current_beta_values") and self.current_beta_values is not None:
            metrics[f"{prefix}beta/mean"] = self.current_beta_values.mean().item()
            
            # 添加pos_beta和neg_beta指标
            if hasattr(self, "current_pos_beta") and self.current_pos_beta is not None:
                metrics[f"{prefix}pos_beta"] = self.current_pos_beta.mean().item()
            
            if hasattr(self, "current_neg_beta") and self.current_neg_beta is not None:
                metrics[f"{prefix}neg_beta"] = self.current_neg_beta.mean().item()
            
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / beta).mean().item()

        return losses.mean(), metrics

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", tuple["torch.Tensor", list["torch.Tensor"]]]:
        r"""Subclass and override to accept extra kwargs."""
        return super().compute_loss(model, inputs, return_outputs)

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        r"""Log `logs` on the various objects watching training, including stored metrics."""
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

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""Compute loss for preference learning."""
        # 计算完整的delta值，考虑参考模型
        if not self.finetuning_args.use_ref_model:
            # 如果不使用参考模型，则简化为policy_chosen_logps - policy_rejected_logps
            self.current_delta_values = policy_chosen_logps - policy_rejected_logps
        else:
            # 按照理论公式计算delta: (π_θ(y_w|x) - π_ref(y_w|x)) - (π_θ(y_l|x) - π_ref(y_l|x))
            if reference_chosen_logps is not None and reference_rejected_logps is not None:
                self.current_delta_values = (policy_chosen_logps - reference_chosen_logps) - (policy_rejected_logps - reference_rejected_logps)
            else:
                # 如果没有参考模型的logps，则使用简化计算
                self.current_delta_values = policy_chosen_logps - policy_rejected_logps
        
        # 根据delta值划分pos_delta和neg_delta样本
        if self.current_delta_values is not None:
            pos_mask = self.current_delta_values > 0  # Δ > 0的样本掩码
            neg_mask = ~pos_mask  # Δ <= 0的样本掩码
            
            if self.use_dynamic_beta and self.current_beta_values is not None:
                # 获取pos_beta和neg_beta
                self.current_pos_beta = self.current_beta_values[pos_mask] if pos_mask.any() else None
                self.current_neg_beta = self.current_beta_values[neg_mask] if neg_mask.any() else None
        
        if not self.finetuning_args.use_ref_model:
            # 使用动态beta值(如果可用)或回退到静态beta
            beta_values = self.current_beta_values if self.use_dynamic_beta and self.current_beta_values is not None else self.beta
            
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps, beta_values)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps, beta_values)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

            # 在reward计算中同样使用动态beta
            chosen_rewards = beta_values * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = beta_values * policy_rejected_logps.to(self.accelerator.device).detach()
        else: # 如果 use_dynamic_beta 为 True，则暂时性使用 self.current_beta_values 计算损失
            tmp_beta = self.beta
            self.beta = self.current_beta_values if self.use_dynamic_beta and self.current_beta_values is not None else self.beta
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            ) # 这个函数使用 self.beta 计算损失
            self.beta = tmp_beta # 恢复 self.beta 的值

        return losses, chosen_rewards, rejected_rewards
