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
import os
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

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma
        
        Trainer.__init__(self, model=model, **kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        # 为 model 新增一个 valuehead 来计算 beta
        self.model.value_head = torch.nn.Sequential(
            torch.nn.Linear(model.config.hidden_size, model.config.hidden_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(model.config.hidden_size, 1),
            torch.nn.Softplus()
        )

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
            # 检查模型是否有value_head属性
            if not hasattr(self.model, "value_head"):
                # 如果没有value_head属性，使用原始方法创建优化器
                self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
                return super().create_optimizer()
            
            # 准备参数组
            if hasattr(self.finetuning_args, "use_galore") and self.finetuning_args.use_galore:
                # 当使用GaLore时，我们需要特殊处理
                from ..trainer_utils import _create_galore_optimizer
                self.optimizer = _create_galore_optimizer(self.model, self.args, self.finetuning_args)
                # 添加value_head参数组
                value_head_params = list(self.model.value_head.parameters())
                if value_head_params:  # 确保有参数
                    self.optimizer.add_param_group({
                        "params": value_head_params, 
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate * 10.0,  # 可选：为value_head使用更高的学习率
                    })
            elif hasattr(self.finetuning_args, "use_apollo") and self.finetuning_args.use_apollo:
                # 当使用Apollo时，我们需要特殊处理
                from ..trainer_utils import _create_apollo_optimizer
                self.optimizer = _create_apollo_optimizer(self.model, self.args, self.finetuning_args)
                # 添加value_head参数组
                value_head_params = list(self.model.value_head.parameters())
                if value_head_params:  # 确保有参数
                    self.optimizer.add_param_group({
                        "params": value_head_params, 
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate * 2.0,  # 可选：为value_head使用更高的学习率
                    })
            else:
                # 标准优化器创建
                optimizer_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
                
                # 获取value_head参数
                value_head_params = list(self.model.value_head.parameters())
                
                # 创建两个参数组：一个用于模型主体参数，一个用于value_head
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() 
                                 if p.requires_grad and not any(vp.data_ptr() == p.data_ptr() for vp in value_head_params)],
                    }
                ]
                
                # 如果value_head有参数，添加专门的参数组
                if value_head_params:
                    optimizer_grouped_parameters.append({
                        "params": value_head_params,
                        "weight_decay": 0.0,  # 不对value_head应用权重衰减
                        "lr": self.args.learning_rate * 10.0,  # 可选：为value_head使用更高的学习率
                    })
                
                # 创建包含所有参数组的优化器
                self.optimizer = optimizer_cls(
                    optimizer_grouped_parameters,
                    **optim_kwargs,
                )

            # 打印参数组信息以便调试
            print("优化器参数组:")
            for i, group in enumerate(self.optimizer.param_groups):
                print(f"参数组 {i}: {len(group['params'])} 个参数, 学习率: {group.get('lr', 'default')}")
        
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

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor", dynamic_beta: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + dynamic_beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor", dynamic_beta: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / dynamic_beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(dynamic_beta * logits)
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
        
        logits = dynamic_beta * (pi_logratios - ref_logratios)  # shape = [batch_size]
        
        if self.label_smoothing > 0:
            # 为标签应用平滑处理
            losses = (
                -self.label_smoothing * F.logsigmoid(-logits) - (1 - self.label_smoothing) * F.logsigmoid(logits)
            )
        else:
            losses = -F.logsigmoid(logits)
        
        chosen_rewards = dynamic_beta * (policy_chosen_logps - reference_chosen_logps).detach() 
        rejected_rewards = dynamic_beta * (policy_rejected_logps - reference_rejected_logps).detach()
        
        return losses, chosen_rewards, rejected_rewards
    
    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """

        # 计算 dynamic beta
        dynamic_beta = self.beta

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
            losses, chosen_rewards, rejected_rewards = self.dpo_loss_with_dynamic_beta(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, dynamic_beta
            )

        return losses, chosen_rewards, rejected_rewards

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """扩展方法，额外返回计算的困惑度"""
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # 避免错误

        outputs = model(**batch, return_dict=True, use_cache=False, output_hidden_states=True) # outputs.keys() = ['logits', 'hidden_states']
        
        all_logits: "torch.Tensor" = outputs.logits.to(torch.float32) # batch.keys() = ['input_ids', 'attention_mask', 'labels']
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"]) # all_logps.shape = [batch_size * 2, seq_len], valid_length.shape = [batch_size * 2]
        
        # 计算prompt的困惑度 - 使用现有的计算，避免额外前向传播
        prompt_perplexity, last_prompt_token_hidden_states = self.calculate_prompt_perplexity_and_last_token_logits(batch, outputs)  # perplexity.shape = [batch_size]
        
        # 计算 value_head 的输出用于计算 dynamic beta
        self.beta = model.value_head(last_prompt_token_hidden_states).squeeze(-1) # input.shape = [batch_size, hidden_size], output.shape = [batch_size]

        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2  # batch_size 是 input_ids.shape[0] 的一半, 因为input_ids是chosen和rejected的拼接, batch_size 是 prompt的batch_size. 
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)

        if self.loss_type in ["ipo", "orpo", "simpo"]:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps, prompt_perplexity, self.beta
        else:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length, prompt_perplexity, self.beta
        
    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
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

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
            prompt_perplexity,
            dynamic_beta
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
        metrics[f"{prefix}beta"] = dynamic_beta.mean().item()
        metrics[f"{prefix}perplexity"] = prompt_perplexity.mean().item()
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()

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


    def calculate_prompt_perplexity_and_last_token_logits(self, batch, outputs):
        """
        使用已有的信息计算prompt的困惑度， last prompt token transformer 的输出用于计算 dynamic beta for each sample.
        """
        # 获取提示部分的mask
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        prompt_mask = (labels == IGNORE_INDEX)
        
        batch_size = input_ids.size(0) // 2
        chosen_input_ids = input_ids[:batch_size]  # 只需计算chosen样本的prompt困惑度
        chosen_prompt_mask = prompt_mask[:batch_size]
        
        # 排除padding位置
        valid_tokens_mask = (chosen_input_ids != self.padding_value) & chosen_prompt_mask
        
        # 已经有logits的情况下直接使用
        assert outputs.logits is not None, "all_logits is None"

        all_logits = outputs.logits[:batch_size]
        chosen_logits = all_logits[:batch_size]
        
        # 只需要prompt部分的logits和对应的input_ids，用于计算下一个token的预测
        # 将token shift：logits预测下一个token
        shifted_logits = chosen_logits[:, :-1, :]  # [batch, seq_len-1, vocab_size]
        shifted_ids = chosen_input_ids[:, 1:]  # [batch, seq_len-1]
        shifted_mask = valid_tokens_mask[:, 1:]  # [batch, seq_len-1]
        
        # 只保留prompt部分和下一个token是prompt的预测
        # 注意：最后一个prompt token预测的是第一个回答token，应排除
        prompt_next_token_mask = shifted_mask & (labels[:batch_size, 1:] == IGNORE_INDEX)
        
        # 计算log softmax
        log_probs = F.log_softmax(shifted_logits, dim=-1)
        
        # 创建安全gather的索引
        gather_ids = shifted_ids.clone()
        valid_gather_mask = (shifted_ids != 0) & prompt_next_token_mask
        gather_ids[~valid_gather_mask] = 1  # 使用一个有效索引
        
        # 获取每个位置对应token的log概率
        token_log_probs = torch.gather(log_probs, -1, gather_ids.unsqueeze(-1)).squeeze(-1)
        
        # 只保留valid位置的log概率
        token_log_probs = token_log_probs * valid_gather_mask.float()
        
        # 计算每个序列有效token数
        seq_lengths = valid_gather_mask.sum(dim=-1).float()
        seq_lengths = torch.max(seq_lengths, torch.ones_like(seq_lengths))  # 避免除零
        
        # 计算平均log概率和困惑度
        avg_log_probs = token_log_probs.sum(dim=-1) / seq_lengths
        perplexity = torch.exp(-avg_log_probs)
        
        # 使用 seq_lengths 计算 last prompt token transformer 的输出 with shape [batch, hidden_size]
        # 将 seq_lengths 转换为整数类型用于索引，并确保在有效范围内
        last_hidden_states = outputs.hidden_states[-1] # shape = [batch, seq_len, hidden_size]
        seq_length_indices = (seq_lengths - 1).long().clamp(0, chosen_logits.size(1) - 1)
        
        # 创建批次索引 [0, 1, 2, ..., batch_size-1]
        batch_indices = torch.arange(batch_size, device=chosen_logits.device)
        
        # 使用高级索引直接获取每个样本最后提示词的logits，完全向量化操作
        last_prompt_token_hidden_states = last_hidden_states[batch_indices, seq_length_indices]
        
        return perplexity, last_prompt_token_hidden_states
        