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
        """
        LEDPO训练器初始化
        
        LEDPO (Learnable Beta DPO) 是标准DPO的扩展，特点是将beta参数设计为可学习参数。
        beta参数控制偏好学习的强度，传统DPO中这是一个固定超参数，而LEDPO中它变为可学习参数。
        
        参数:
            model: 策略模型，将被优化
            ref_model: 参考模型，用于计算KL散度约束
            finetuning_args: 微调参数
            processor: 可选的处理器组件
            disable_dropout: 是否禁用dropout（评估/推理时通常设为True）
            **kwargs: 额外参数
        """
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        # 禁用dropout以提高训练稳定性
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        # 初始化各种训练参数和设置
        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"  # 使用反向KL散度作为正则化
        self.reference_free = False
        self.use_dpo_data_collator = True  # 避免warning的技巧
        self.generate_during_eval = False   # 评估时禁用生成
        self.label_pad_token_id = IGNORE_INDEX  # 标签填充token的ID
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # DPO超参数设置
        self.beta = finetuning_args.pref_beta  # 基础beta值 - 从配置中获取初始值
        self.loss_type = finetuning_args.pref_loss  # 损失类型（dpo/orpo/simpo等）
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing  # 标签平滑参数
        self.simpo_gamma = finetuning_args.simpo_gamma  # SimPO损失的gamma参数
        
        # LEDPO核心: 创建可学习的beta_scale参数
        # 初始化为1.0，使训练开始时dynamic_beta = beta
        # requires_grad=True使其参与梯度更新
        beta_scale_param = torch.nn.Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
        
        # 调用标准Trainer初始化
        Trainer.__init__(self, model=model, **kwargs)
        
        # 将beta_scale设置为模型的属性
        # 必须在Trainer初始化后设置，确保它正确地添加到模型参数中
        self.model.beta_scale = beta_scale_param
        
        self.model_accepts_loss_kwargs = False  # 覆盖trainer的默认行为
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # 移除ref模型的gc警告

        # 准备参考模型（如果提供）
        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # 量化模型已经在正确设备上
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

    def get_dynamic_beta(self):
        """
        获取动态beta值
        
        LEDPO的核心创新：通过可学习的beta_scale参数动态调整beta值。
        随着训练进行，模型会自动学习最优的beta_scale值，从而使得beta更加适应当前训练状态。
        """
        # 使用可学习的beta_scale调整基础beta值
        # beta_scale会在训练过程中通过梯度下降自动更新
        dynamic_beta = self.beta * self.model.beta_scale
        return dynamic_beta

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        计算ORPO(Odds Ratio Policy Optimization)损失函数
        
        ORPO损失结合了SFT损失和偏好优化损失，利用log odds ratio来构建。
        这里使用动态beta来控制偏好学习的强度。
        
        参数:
            chosen_logps: 偏好数据中较好回答的对数概率
            rejected_logps: 偏好数据中较差回答的对数概率
            
        返回:
            计算得到的ORPO损失
        """
        # 获取当前的动态beta值
        dynamic_beta = self.get_dynamic_beta()
        
        # 计算log odds ratio
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        
        # SFT损失部分（极大化chosen样本概率）
        sft_loss = -chosen_logps
        
        # Odds Ratio损失部分
        odds_ratio_loss = -F.logsigmoid(log_odds)
        
        # 组合损失，使用动态beta控制偏好学习的强度
        orpo_loss = sft_loss + dynamic_beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        计算SimPO(Simple Policy Optimization)损失函数
        
        SimPO是一种简化的偏好优化方法，使用动态beta来控制偏好学习的强度。
        
        参数:
            chosen_logps: 偏好数据中较好回答的对数概率
            rejected_logps: 偏好数据中较差回答的对数概率
            
        返回:
            计算得到的SimPO损失
        """
        # 获取当前的动态beta值
        dynamic_beta = self.get_dynamic_beta()
        
        # 计算策略对数比率
        pi_logratios = chosen_logps - rejected_logps
        
        # 计算gamma对数比率
        gamma_logratios = self.simpo_gamma / dynamic_beta
        
        # 计算最终logits
        logits = pi_logratios - gamma_logratios
        
        # 计算SimPO损失，使用动态beta调整学习强度
        simpo_loss = -F.logsigmoid(dynamic_beta * logits)
        return simpo_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        计算偏好学习的损失函数
        
        根据设置的损失类型和是否使用参考模型，选择不同的损失计算方法。
        所有方法都使用动态beta来控制偏好学习的强度。
        
        参数:
            policy_chosen_logps: 策略模型对较好回答的对数概率
            policy_rejected_logps: 策略模型对较差回答的对数概率
            reference_chosen_logps: 参考模型对较好回答的对数概率
            reference_rejected_logps: 参考模型对较差回答的对数概率
            
        返回:
            (损失, 较好回答的奖励, 较差回答的奖励)
        """
        # 获取当前的动态beta值
        dynamic_beta = self.get_dynamic_beta()
        
        # 根据是否使用参考模型选择不同的损失计算方法
        if not self.finetuning_args.use_ref_model:
            # 不使用参考模型时，根据损失类型选择
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

            # 计算奖励值（用于记录和可视化）
            chosen_rewards = dynamic_beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = dynamic_beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            # 使用参考模型时，调用DPO损失计算方法
            # 注意这也使用动态beta
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )

        return losses, chosen_rewards, rejected_rewards

    def dpo_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: "torch.Tensor",
        reference_rejected_logps: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        使用可学习beta计算DPO损失
        
        标准DPO损失的LEDPO变体，使用动态beta参数控制偏好学习的强度。
        DPO通过最大化策略模型和参考模型之间的对数比率差来优化策略。
        
        参数:
            policy_chosen_logps: 策略模型对较好回答的对数概率
            policy_rejected_logps: 策略模型对较差回答的对数概率
            reference_chosen_logps: 参考模型对较好回答的对数概率
            reference_rejected_logps: 参考模型对较差回答的对数概率
            
        返回:
            (DPO损失, 较好回答的奖励, 较差回答的奖励)
        """
        # 获取当前的动态beta值
        dynamic_beta = self.get_dynamic_beta()
        
        # 计算策略模型的对数比率
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        # 计算参考模型的对数比率
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        # 计算最终logits（策略和参考模型的对数比率差）
        logits = pi_logratios - ref_logratios
        
        # DPO标准损失函数计算
        if self.label_smoothing > 0:
            # 使用标签平滑，alpha确定平滑程度
            alpha = self.label_smoothing
            # 平滑后的DPO损失
            losses = -alpha * F.logsigmoid(dynamic_beta * logits) - (1 - alpha) * F.logsigmoid(-dynamic_beta * logits)
        else:
            # 标准DPO损失，使用动态beta调整学习强度
            losses = -F.logsigmoid(dynamic_beta * logits)
        
        # 计算奖励值（用于记录和可视化）
        chosen_rewards = dynamic_beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = dynamic_beta * (policy_rejected_logps - reference_rejected_logps).detach()
        
        return losses, chosen_rewards, rejected_rewards

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # avoid error

        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)

        if self.loss_type in ["ipo", "orpo", "simpo"]:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps
        else:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length

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
            policy_chosen_avg_logps,
        ) = self.concatenated_forward(model, batch)

        # 计算动态beta值
        dynamic_beta = self.get_dynamic_beta()
        metrics["beta_scale"] = self.model.beta_scale.detach()
        metrics["dynamic_beta"] = dynamic_beta.detach()

        if self.loss_type == "ftx" and self.ftx_gamma > 0:
            sft_loss = self.ftx_gamma * -policy_chosen_logps.mean()
            metrics["sft_loss"] = sft_loss.detach()

        if (train_eval == "train" and self._precomputed_train_ref_log_probs) or (
            train_eval == "eval" and self._precomputed_eval_ref_log_probs
        ):
            batch_size = policy_chosen_logps.shape[0]
            if train_eval == "train":
                if self.precomputed_train_reference_log_probs["chosen"].shape[0] < batch_size:
                    batch_size = self.precomputed_train_reference_log_probs["chosen"].shape[0]
            else:
                if self.precomputed_eval_reference_log_probs["chosen"].shape[0] < batch_size:
                    batch_size = self.precomputed_eval_reference_log_probs["chosen"].shape[0]

            if train_eval == "train":
                reference_chosen_logps = self.precomputed_train_reference_log_probs["chosen"][:batch_size].to(
                    self.accelerator.device
                )
                reference_rejected_logps = self.precomputed_train_reference_log_probs["rejected"][:batch_size].to(
                    self.accelerator.device
                )
            else:
                reference_chosen_logps = self.precomputed_eval_reference_log_probs["chosen"][:batch_size].to(
                    self.accelerator.device
                )
                reference_rejected_logps = self.precomputed_eval_reference_log_probs["rejected"][:batch_size].to(
                    self.accelerator.device
                )

            policy_chosen_logps = policy_chosen_logps[:batch_size]
            policy_rejected_logps = policy_rejected_logps[:batch_size]
            policy_chosen_logits = policy_chosen_logits[:batch_size]
            policy_rejected_logits = policy_rejected_logits[:batch_size]
            policy_chosen_avg_logps = policy_chosen_avg_logps[:batch_size]
        else:
            reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)

        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
        )

        if self.loss_type == "ftx" and self.ftx_gamma > 0:
            loss = losses.mean() + sft_loss
        else:
            loss = losses.mean()

        metrics["chosen_rewards"] = chosen_rewards.detach().mean()
        metrics["rejected_rewards"] = rejected_rewards.detach().mean()
        metrics["margins"] = (chosen_rewards - rejected_rewards).detach().mean()
        metrics["accuracies"] = (chosen_rewards > rejected_rewards).detach().float().mean()
        # metrics["policy_rejected_logps"] = policy_rejected_logps.detach().mean()
        # metrics["policy_chosen_logps"] = policy_chosen_logps.detach().mean()
        # metrics["policy_chosen_avg_logps"] = policy_chosen_avg_logps.detach().mean()

        return loss, metrics

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        r"""
        Overrides the compute_loss method of the trainer to compute the DPO loss.
        """
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        if not is_transformers_version_greater_than("4.41.0"):
            self.log(metrics)
        else:
            # metrics will be logged in log method
            self._stored_metrics["train"].update(metrics)
        return loss

    @override
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        r"""
        Overrides the log method of the trainer to capture the loss metrics.
        """
        if not is_transformers_version_greater_than("4.41.0"):
            super().log(logs, *args, **kwargs)
            return

        if "epoch" in logs and hasattr(self, "_stored_metrics"):
            prefix = "train"
            stored_metrics = self._stored_metrics.pop(prefix, {})
            if len(stored_metrics) > 0:
                for metric_name, metrics_values in stored_metrics.items():
                    if len(metrics_values) > 0:
                        logs[f"{prefix}_{metric_name}"] = torch.tensor(metrics_values).mean().item()
        super().log(logs, *args, **kwargs)
