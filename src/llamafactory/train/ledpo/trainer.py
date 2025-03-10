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
        self.beta = finetuning_args.pref_beta  # 设置初始beta值，从配置文件读取
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma
        self.use_beta_head = getattr(finetuning_args, "use_beta_head", True)  # 决定是否使用value_head动态计算beta
        self.beta_scale = getattr(finetuning_args, "pref_beta_scale", 1.0)  # 获取beta值的缩放因子，默认为1.0
        self.beta_head_activation_fn = getattr(finetuning_args, "beta_head_activation_fn", "sigmoid").lower()
        self.value_head_lr_multiplier = getattr(finetuning_args, "value_head_lr_multiplier", 100.0)  # 获取value_head的学习率倍率，默认为100.0
        Trainer.__init__(self, model=model, **kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        # 根据use_beta_head参数决定是否添加value_head
        if self.use_beta_head:
            # 为model新增一个value_head网络，用于从模型隐藏状态动态计算beta值
            # 重要：这是动态beta计算的核心结构，如果要调试动态beta的计算，应该在这里设置断点
            # 预期行为：value_head接收隐藏状态，输出一个正数作为beta值
            # 网络结构：Linear -> Sigmoid -> Linear -> 激活函数
            
            # 根据参数选择激活函数
            if self.beta_head_activation_fn == "softplus":
                activation_fn = torch.nn.Softplus()
            elif self.beta_head_activation_fn == "relu":
                activation_fn = torch.nn.ReLU()
            elif self.beta_head_activation_fn == "sigmoid": 
                activation_fn = torch.nn.Sigmoid()
            else:
                raise ValueError(f"Unknown activation function: {self.beta_head_activation_fn}")
                
            # 简化后的单层ValueHead网络
            self.model.value_head = torch.nn.Sequential(
                # 移除第一层网络
                torch.nn.Linear(model.config.hidden_size, model.config.hidden_size),
                torch.nn.Sigmoid(), # 做完实验后恢复
                # 直接使用单层映射
                torch.nn.Linear(model.config.hidden_size, 1),
                activation_fn  # 使用选择的激活函数
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
        
        
        self.value_head_lr = self.args.learning_rate * self.value_head_lr_multiplier  # 计算value_head的学习率

        if self.optimizer is None:
            # 检查模型是否有value_head属性
            if not hasattr(self.model, "value_head"):
                # 如果没有value_head属性，使用原始方法创建优化器
                self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
                return super().create_optimizer()
            
            # 获取value_head参数
            value_head_params = list(self.model.value_head.parameters())
            
            # 检查冻结设置，不允许同时冻结 value_head 和 policy
            if self.finetuning_args.freeze_value_head and self.finetuning_args.freeze_policy:
                raise ValueError("不能同时设置 freeze_value_head=True 和 freeze_policy=True")
            
            # 应用冻结设置
            if self.finetuning_args.freeze_value_head:
                print("【冻结设置】冻结 ValueHead，只训练 Policy")
                for p in value_head_params:
                    p.requires_grad = False
            
            if self.finetuning_args.freeze_policy:
                print("【冻结设置】冻结 Policy，只训练 ValueHead")
                # 获取所有非value_head参数
                policy_params = [p for n, p in self.model.named_parameters() 
                              if p.requires_grad and not any(vp.data_ptr() == p.data_ptr() for vp in value_head_params)]
                # 冻结所有非value_head参数
                for p in policy_params:
                    p.requires_grad = False
                    
            # 准备参数组
            if hasattr(self.finetuning_args, "use_galore") and self.finetuning_args.use_galore:
                # 当使用GaLore时，我们需要特殊处理
                from ..trainer_utils import _create_galore_optimizer
                self.optimizer = _create_galore_optimizer(self.model, self.args, self.finetuning_args)
                # 添加value_head参数组
                value_head_params = [p for p in value_head_params if p.requires_grad]  # 只添加需要训练的参数
                if value_head_params:  # 确保有参数
                    # 获取value_head学习率倍率，从finetuning_args获取而不是args
                    # 添加监控: 打印实际使用的学习率倍率
                    print(f"【监控】ValueHead学习率: {self.value_head_lr})")
                    self.optimizer.add_param_group({
                        "params": value_head_params, 
                        "weight_decay": 0.0,
                        "lr": self.value_head_lr,  # 使用可配置的学习率倍率
                    })
            elif hasattr(self.finetuning_args, "use_apollo") and self.finetuning_args.use_apollo:
                # 当使用Apollo时，我们需要特殊处理
                from ..trainer_utils import _create_apollo_optimizer
                self.optimizer = _create_apollo_optimizer(self.model, self.args, self.finetuning_args)
                # 添加value_head参数组
                value_head_params = [p for p in value_head_params if p.requires_grad]  # 只添加需要训练的参数
                if value_head_params:  # 确保有参数
                    # 获取value_head学习率倍率，从finetuning_args获取而不是args
                    # 添加监控: 打印实际使用的学习率倍率
                    print(f"【监控】ValueHead学习率: {self.value_head_lr})")
                    self.optimizer.add_param_group({
                        "params": value_head_params, 
                        "weight_decay": 0.0,
                        "lr": self.value_head_lr,  # 使用可配置的学习率倍率
                    })
            else:
                # 标准优化器创建
                optimizer_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
                
                # 创建两个参数组：一个用于模型主体参数，一个用于value_head
                optimizer_grouped_parameters = []
                
                # 获取所有非value_head需要训练的参数
                policy_params = [p for n, p in self.model.named_parameters() 
                               if p.requires_grad and not any(vp.data_ptr() == p.data_ptr() for vp in value_head_params)]
                
                if policy_params:  # 如果有需要训练的policy参数
                    optimizer_grouped_parameters.append({
                        "params": policy_params,
                    })
                
                # 如果value_head有需要训练的参数，添加专门的参数组
                value_head_params = [p for p in value_head_params if p.requires_grad]
                if value_head_params:
                    # 添加监控: 打印实际使用的学习率倍率
                    print(f"【监控】ValueHead学习率: {self.value_head_lr})")
                    optimizer_grouped_parameters.append({
                        "params": value_head_params,
                        "weight_decay": 0.0,  # 不对value_head应用权重衰减
                        "lr": self.value_head_lr,  # 使用可配置的学习率倍率
                    })
                
                # 创建包含所有参数组的优化器
                self.optimizer = optimizer_cls(
                    optimizer_grouped_parameters,
                    **optim_kwargs,
                )

            # 打印参数组信息以便调试
            print("优化器参数组:")
            trainable_params_count = 0
            for i, group in enumerate(self.optimizer.param_groups):
                group_params_count = sum(p.numel() for p in group['params'])
                trainable_params_count += group_params_count
                print(f"参数组 {i}: {len(group['params'])} 个参数张量, 共 {group_params_count} 个参数, 学习率: {group.get('lr', 'default')}")
            
            # 打印可训练参数总数与比例
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"模型总参数: {total_params}, 可训练参数: {trainable_params_count}, 比例: {trainable_params_count/total_params:.6f}")

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
        # 断点位置4：检查进入损失计算前的beta值
        # 检查项：
        # - self.beta的值和形状，特别是不同样本之间beta的分布和变化
        # - 如果使用动态beta，观察self.beta是否有明显的变化趋势或模式
        # - 验证beta值是否与之前计算的值一致
        
        dynamic_beta = self.beta  # 将self.beta赋值给dynamic_beta，无论是静态值还是动态计算的值

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
        
        # 断点位置5：检查计算后的loss和rewards
        # 检查项：
        # - 损失值losses的分布
        # - chosen_rewards和rejected_rewards的分布
        # - beta与losses的相关性（beta大的样本是否有不同的损失特征）

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
        
        # 断点位置1：检查last_prompt_token_hidden_states的形状和值
        # 预期：last_prompt_token_hidden_states应该是[batch_size, hidden_size]的张量，表示每个样本最后一个prompt token的隐藏状态
        
        # 根据use_beta_head参数决定是否使用value_head计算动态beta
        if self.use_beta_head:
            # 断点位置2：计算动态beta之前，检查输入值
            # 检查项：
            # - last_prompt_token_hidden_states的形状和值
            # - model.value_head的结构
            # - self.beta的当前值（将被更新）
            
            # 计算动态beta：将最后一个prompt token的隐藏状态输入到value_head中得到beta值
            # print(f"hidden states: {last_prompt_token_hidden_states.shape}")  # 调试用：打印隐藏状态的形状
            self.beta_head_output = model.value_head(last_prompt_token_hidden_states).squeeze(-1) # input.shape = [batch_size, hidden_size], output.shape = [batch_size]
            
            # 应用beta_scale缩放因子
            self.beta = self.beta_head_output * self.beta_scale
            
            # 打印调试信息
            # print(f"dynamic beta: {self.beta}")  # 调试用：打印动态beta值
        # 如果不使用value_head，则self.beta保持为常量值（初始化时的finetuning_args.pref_beta）

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

        if self.ref_model is None: # 在没有单独的参考模型时，它通过禁用当前模型的LoRA适配器，将"原始模型"(没有微调部分)作为参考模型使用。
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
        # 断点位置6：批次开始前检查beta状态
        # 检查项：
        # - self.beta的初始值（如果是静态beta）或上一批次计算的值（如果是动态beta）
        # - 如果是训练过程中的多个批次，观察beta值是否有变化趋势
        
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
        # 记录当前批次的beta值
        # 如果使用动态beta，则记录batch内所有样本beta的平均值；否则记录静态beta值
        metrics[f"{prefix}beta"] = dynamic_beta.mean().item() if self.use_beta_head else self.beta
        
        # 记录beta_head_output的统计信息（如果使用动态beta）
        if self.use_beta_head and hasattr(self, 'beta_head_output'):
            metrics[f"{prefix}beta_head_output/mean"] = self.beta_head_output.mean().item()
            metrics[f"{prefix}beta_head_output/min"] = self.beta_head_output.min().item()
            metrics[f"{prefix}beta_head_output/max"] = self.beta_head_output.max().item()
            
        metrics[f"{prefix}perplexity"] = prompt_perplexity.mean().item()
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()

        # 断点位置7：批次结束时检查beta和相关指标
        # 检查项：
        # - dynamic_beta的最终值和分布
        # - metrics中记录的beta值
        # - beta和perplexity之间是否有相关性
        # - beta和rewards之间的关系
        
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
        计算prompt的困惑度及最后一个token的hidden states
        这个方法非常关键，它计算用于动态beta的隐藏状态输入
        """
        # 断点位置8：检查困惑度和隐藏状态计算过程的起始点
        # 检查项：
        # - batch的结构和内容
        # - outputs的结构和内容（特别是hidden_states）
        
        # 获取提示部分的mask
        labels = batch["labels"]  # [batch_size, seq_len] LLaMA-Factory使用动态填充(dynamic padding)策略，只填充到当前批次中最长样本的长度, 所以不一定等于 cutoff_len
        input_ids = batch["input_ids"]  # [batch_size, seq_len]
        prompt_mask = (labels == IGNORE_INDEX)  # [batch_size, seq_len]  
        
        batch_size = input_ids.size(0) // 2  # 计算chosen样本的prompt困惑度
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
        
        # 断点位置9：检查困惑度计算结果
        # 检查项：
        # - perplexity的值和分布
        # - 如果perplexity与beta计算有关，验证perplexity的合理性
        
        # 使用 seq_lengths 计算 last prompt token transformer 的输出
        # 这是动态beta计算的关键输入！
        last_hidden_states = outputs.hidden_states[-1] # shape = [batch, seq_len, hidden_size]
        seq_length_indices = (seq_lengths - 1).long().clamp(0, chosen_logits.size(1) - 1)
        
        # 创建批次索引 [0, 1, 2, ..., batch_size-1]
        batch_indices = torch.arange(batch_size, device=chosen_logits.device)
        
        # 使用高级索引直接获取每个样本最后提示词的隐藏状态
        last_prompt_token_hidden_states = last_hidden_states[batch_indices, seq_length_indices]
        
        # 断点位置10：检查用于计算beta的隐藏状态
        # 检查项：
        # - last_prompt_token_hidden_states的形状和值
        # - seq_length_indices是否正确识别了最后的prompt token位置
        # - last_prompt_token_hidden_states与perplexity之间是否有联系
        
        return perplexity, last_prompt_token_hidden_states
        