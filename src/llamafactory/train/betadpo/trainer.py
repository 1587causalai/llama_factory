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
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

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


class CustomBetaDPOTrainer(DPOTrainer):
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

        # betadpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma
        
        # BetaDPO特定参数
        self.beta_strategy = getattr(finetuning_args, "beta_strategy", "adaptive")
        self.beta_min = getattr(finetuning_args, "beta_min", 0.1)
        self.beta_max = getattr(finetuning_args, "beta_max", 10.0)

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

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

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
    
    def calculate_adaptive_beta(self, chosen_logps, rejected_logps):
        """
        计算自适应的beta值
        策略:
        - 'adaptive': 根据logps的差异动态调整beta值
        - 'exponential': 指数增长的beta值
        - 'cosine': 余弦调度的beta值
        """
        # 默认使用基础beta值
        if self.beta_strategy == "constant" or self.beta_strategy is None:
            return self.beta
        
        # 获取当前训练步骤
        current_step = self.state.global_step
        total_steps = self.state.max_steps
        
        if self.beta_strategy == "adaptive":
            # 根据chosen和rejected的log概率差异自适应调整beta
            logp_diff = torch.abs(chosen_logps - rejected_logps).mean().item()
            # 使用sigmoid函数将差异映射到beta_min和beta_max之间
            normalized_diff = 1.0 / (1.0 + torch.exp(-logp_diff + 2.0))  # 中心在2.0处的sigmoid
            beta_value = self.beta_min + (self.beta_max - self.beta_min) * normalized_diff
            return beta_value
            
        elif self.beta_strategy == "exponential":
            # 指数增长的beta值
            progress = min(1.0, current_step / total_steps)
            beta_value = self.beta_min * (self.beta_max / self.beta_min) ** progress
            return beta_value
            
        elif self.beta_strategy == "cosine":
            # 余弦调度的beta值
            progress = min(1.0, current_step / total_steps)
            beta_value = self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (1 + torch.cos(torch.tensor(progress * torch.pi)).item())
            return beta_value
            
        # 默认返回constant beta
        return self.beta

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor", beta: float = None) -> "torch.Tensor":
        """使用自适应beta计算ORPO损失"""
        beta_value = beta if beta is not None else self.calculate_adaptive_beta(chosen_logps, rejected_logps)
        
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + beta_value * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor", beta: float = None) -> "torch.Tensor":
        """使用自适应beta计算SimPO损失"""
        beta_value = beta if beta is not None else self.calculate_adaptive_beta(chosen_logps, rejected_logps)
        
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / beta_value
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(beta_value * logits)
        return simpo_loss

    def dpo_loss_with_adaptive_beta(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: "torch.Tensor",
        reference_rejected_logps: "torch.Tensor",
        beta: float = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        实现支持自适应beta的标准DPO损失
        """
        beta_value = beta if beta is not None else self.calculate_adaptive_beta(policy_chosen_logps, policy_rejected_logps)
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        logits = beta_value * (pi_logratios - ref_logratios)
        
        if self.label_smoothing > 0:
            # 为标签应用平滑处理
            losses = (
                -self.label_smoothing * F.logsigmoid(-logits) - (1 - self.label_smoothing) * F.logsigmoid(logits)
            )
        else:
            losses = -F.logsigmoid(logits)
        
        chosen_rewards = beta_value * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta_value * (policy_rejected_logps - reference_rejected_logps).detach()
        
        return losses, chosen_rewards, rejected_rewards

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        计算偏好学习的损失，使用自适应beta
        """
        # 计算自适应的beta值
        beta_value = self.calculate_adaptive_beta(policy_chosen_logps, policy_rejected_logps)
        
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps, beta_value)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps, beta_value)
            else:
                raise NotImplementedError(f"未知的损失类型: {self.loss_type}.")

            chosen_rewards = beta_value * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = beta_value * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            # 标准DPO逻辑但使用自适应beta
            losses, chosen_rewards, rejected_rewards = self.dpo_loss_with_adaptive_beta(
                policy_chosen_logps, policy_rejected_logps, 
                reference_chosen_logps, reference_rejected_logps,
                beta_value
            )

        # 记录当前使用的beta值 - 修复：直接添加到metrics字典，而不是调用log方法
        if self.is_world_process_zero():
            self._stored_metrics["train"]["beta/current_value"].append(float(beta_value))
        
        return losses, chosen_rewards, rejected_rewards

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """返回模型的forward结果"""
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # 避免错误

        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """计算参考模型的log概率，与标准DPO相同"""
        if not self.finetuning_args.use_ref_model:
            return None, None

        with torch.no_grad():
            all_logits = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
            all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
            
            if self.loss_type in ["ipo", "orpo", "simpo"]:
                all_logps = all_logps / valid_length

            batch_size = batch["input_ids"].size(0) // 2  # 因为是成对的数据
            reference_chosen_logps, reference_rejected_logps = all_logps.split(batch_size, dim=0)
            
        return reference_chosen_logps, reference_rejected_logps

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        """计算批次的损失和指标"""
        metrics = {}
        
        # Forward pass
        chosen_logps, rejected_logps, chosen_logits, rejected_logits = self.concatenated_forward(model, batch)

        # 获取参考模型的log概率
        batch_ref = {k: v for k, v in batch.items()}  # 创建batch的副本
        
        if self._precomputed_train_ref_log_probs and train_eval == "train":
            # 如果预计算了训练集的参考log概率
            reference_chosen_logps, reference_rejected_logps = batch.get("reference_chosen_logps", None), batch.get(
                "reference_rejected_logps", None
            )
        elif self._precomputed_eval_ref_log_probs and train_eval == "eval":
            # 如果预计算了评估集的参考log概率
            reference_chosen_logps, reference_rejected_logps = batch.get("reference_chosen_logps", None), batch.get(
                "reference_rejected_logps", None
            )
        else:
            with torch.no_grad():
                if self.ref_model is not None:
                    self.ref_model.eval()
                    # 使用参考模型计算log概率
                    reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(
                        self.ref_model, batch_ref
                    )
                else:
                    # 重用当前模型计算参考log概率
                    reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch_ref)

        # 计算偏好损失
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
        )
        
        # 添加监督学习损失（如果需要）
        if self.ftx_gamma > 0:
            batch_size = chosen_logits.shape[0]
            sft_chosen_logits = chosen_logits
            sft_labels = batch["labels"][:batch_size]
            
            if not self.is_encoder_decoder:
                sft_labels = sft_labels[:, 1:].contiguous()
                sft_chosen_logits = sft_chosen_logits[:, :-1, :].contiguous()
            
            sup_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.label_pad_token_id)
            sup_loss = sup_loss_fct(sft_chosen_logits.view(-1, sft_chosen_logits.size(-1)), sft_labels.view(-1))
            
            combined_loss = (1 - self.ftx_gamma) * losses.mean() + self.ftx_gamma * sup_loss
        else:
            combined_loss = losses.mean()

        # 添加指标
        metrics["rewards/chosen"] = chosen_rewards.mean().item()
        metrics["rewards/rejected"] = rejected_rewards.mean().item()
        metrics["rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics["rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics["logps/chosen"] = chosen_logps.mean().item()
        metrics["logps/rejected"] = rejected_logps.mean().item()
        metrics["logps/margins"] = (chosen_logps - rejected_logps).mean().item()
        
        if reference_chosen_logps is not None and reference_rejected_logps is not None:
            metrics["logps/reference/chosen"] = reference_chosen_logps.mean().item()
            metrics["logps/reference/rejected"] = reference_rejected_logps.mean().item()
            metrics["logps/reference/margins"] = (reference_chosen_logps - reference_rejected_logps).mean().item()

        return combined_loss, metrics

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        """
        计算损失的主函数
        """
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        
        # 只在主进程上更新指标
        if self.is_world_process_zero():
            self._stored_metrics["train"].update(metrics)
        
        return loss

    @override
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        记录日志
        """
        # 添加当前batch的指标
        for k, v in self._stored_metrics["train"].items():
            if isinstance(v, list) and len(v) > 0:
                logs[k] = v[-1]
                
        if kwargs.get("clear_metrics", True):
            self._stored_metrics["train"] = defaultdict(list)
            
        super().log(logs, *args, **kwargs) 