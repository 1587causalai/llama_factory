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

from ...extras.constants import IGNORE_INDEX # 注意: 如果你的代码环境中没有 ...extras.constants，需要根据你的实际情况调整 IGNORE_INDEX 的获取方式，或者直接替换为 -100
from ...extras.packages import is_transformers_version_greater_than # 注意: 同样需要根据实际情况调整 ...extras.packages
from ..callbacks import SaveProcessorCallback # 注意: 同样需要根据实际情况调整 ...callbacks
from ..trainer_utils import get_batch_logps, nested_detach, create_custom_optimizer, create_custom_scheduler # 注意: 同样需要根据实际情况调整 ...trainer_utils
from .beta_head import BetaHead, BetaHeadConfig  # 确保正确导入 BetaHead 和 BetaHeadConfig

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from trl import TrainingArguments # 假设 FinetuningArguments 可以替换为 trl.TrainingArguments，或者你需要自定义一个类似的配置类

    from ...hparams import FinetuningArguments #  如果 FinetuningArguments 仍然被使用，需要保留

class LearnableBetaDPOTrainer(DPOTrainer): #  直接继承 DPOTrainer
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
        self.current_batch = None  # 添加 current_batch 属性

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta  # 初始 beta 值，用于非动态 beta 的场景
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        # Learnable Beta DPO 参数初始化
        beta_head_config = BetaHeadConfig(
            hidden_size=model.config.hidden_size,
            nn_type=finetuning_args.beta_head_type,
            epsilon=finetuning_args.beta_head_epsilon
        )
        self.beta_head = BetaHead(beta_head_config)

        Trainer.__init__(self, model=model, **kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        # 将 beta_head 移动到正确的设备上并准备好
        self.beta_head = self.accelerator.prepare_model(self.beta_head, evaluation_mode=False)

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

    @override
    def get_dynamic_beta_and_ppl(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        model_outputs: Optional[Dict[str, "torch.Tensor"]] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """计算动态 beta 值和 PPL，保留梯度以训练 beta head
        
        Args:
            input_ids: 输入的 token ids [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            model_outputs: 可选的模型输出，如果提供则复用其计算结果
            
        Returns:
            beta: 动态计算的 beta 值 [batch_size]
            ppl: 计算的困惑度值 [batch_size]
        """
        # 1. 获取模型输出
        if model_outputs is None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        else:
            outputs = model_outputs
            
        # 2. 找到 assistant 标记的位置作为 prompt 结束位置
        assistant_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>assistant")
        batch_size = input_ids.size(0)
        prompt_lengths = []
        for i in range(batch_size):
            assistant_pos = (input_ids[i] == assistant_token_id).nonzero()
            if len(assistant_pos) > 0:
                prompt_lengths.append(assistant_pos[-1].item())
            else:
                prompt_lengths.append(input_ids.size(1))
        
        # 3. 提取正确的 context embedding
        prompt_end_indices = torch.tensor(prompt_lengths, device=input_ids.device) - 1
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        context_embedding = outputs.hidden_states[-1][batch_indices, prompt_end_indices, :]
        
        # 4. 计算 prompt 部分的 PPL
        logits = outputs.logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
        labels = input_ids[:, 1:]  # [batch_size, seq_len-1]
        mask = attention_mask[:, 1:].float()  # [batch_size, seq_len-1]
        
        # 计算每个位置的 loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction='none'
        ).reshape_as(labels)  # [batch_size, seq_len-1]
        
        # 只使用 prompt 部分计算 PPL
        prompt_ppls = []
        for i in range(batch_size):
            prompt_len = prompt_lengths[i] - 1  # -1 因为我们是用当前token预测下一个
            if prompt_len > 0:
                prompt_loss = loss[i, :prompt_len]
                prompt_mask = mask[i, :prompt_len]
                # 确保我们有足够的有效 token
                if prompt_mask.sum() > 0:
                    avg_prompt_loss = (prompt_loss * prompt_mask).sum() / prompt_mask.sum()
                    prompt_ppls.append(torch.exp(avg_prompt_loss))
                else:
                    # 如果没有有效 token，使用一个默认值
                    prompt_ppls.append(torch.tensor(1.0, device=loss.device))
            else:
                # 如果找不到 prompt 长度，使用一个默认值
                prompt_ppls.append(torch.tensor(1.0, device=loss.device))
        
        ppl = torch.stack(prompt_ppls)
        
        # 5. 计算动态 beta 值
        beta = self.beta_head.get_dynamic_beta(context_embedding, ppl)
        
        return beta, ppl

    @override
    def concatenated_forward(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """与 DPOTrainer 保持一致的前向传播方法，同时计算动态 beta 值
        
        Returns:
            chosen_logps: chosen 样本的对数概率 [batch_size/2]
            rejected_logps: rejected 样本的对数概率 [batch_size/2]
            chosen_logits: rejected 样本的 logits [batch_size/2, seq_len, vocab_size]
            rejected_logits: rejected 样本的 logits [batch_size/2, seq_len, vocab_size]
            chosen_logps_avg: chosen 样本的平均对数概率 [batch_size/2]
            betas: 每个输入的动态 beta 值 [batch_size/2] (每对 chosen/rejected 共享同一个 beta)
            ppls: 每个输入的困惑度值 [batch_size/2]
        """
        if self.finetuning_args is not None and self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)

        # 获取模型输出
        outputs = model(
            **batch,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False
        )
        
        last_layer_hidden_state = outputs.hidden_states[-1]  # 手动选择最后一层
        all_logits = outputs.logits


        # 计算 log probabilities
        all_logits = outputs.logits.to(torch.float16)  # [batch_size, seq_len, vocab_size]
        all_logps, valid_length = get_batch_logps(
            logits=all_logits,
            labels=batch["labels"]
        )

        batch_size = batch["input_ids"].size(0)
        half_batch_size = batch_size // 2

        # 注意：我们只需要处理前半部分的数据来计算 beta，因为后半部分是相同输入的不同回答
        input_ids = batch["input_ids"][:half_batch_size]  # 只取前半部分，即 chosen 部分的输入
        attention_mask = batch["attention_mask"][:half_batch_size]

        # 计算每个位置的 loss 用于 PPL（只对前半部分计算）
        chosen_logits = all_logits[:half_batch_size]  # 只使用前半部分的 logits
        
        # 计算每个位置的 loss
        loss_per_token = F.cross_entropy(
            chosen_logits[:, :-1].reshape(-1, chosen_logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
            reduction='none'
        ).reshape(half_batch_size, -1)  # [batch_size//2, seq_len-1]
        
        # 使用 attention mask 来确定有效的 token 位置
        valid_mask = attention_mask[:, 1:].float()  # [batch_size//2, seq_len-1]
        
        # 计算每个样本的平均 loss
        masked_loss = loss_per_token * valid_mask
        token_counts = valid_mask.sum(dim=1)  # 每个样本实际的 token 数量
        
        # 处理 token 数量为 0 的情况
        valid_samples = token_counts > 0
        avg_losses = torch.zeros_like(token_counts)
        avg_losses[valid_samples] = (masked_loss[valid_samples].sum(dim=1) / token_counts[valid_samples])
        
        # 计算困惑度
        ppls = torch.exp(avg_losses)
        
        # 获取最后一层的隐状态
        last_hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # 使用最后一个非填充位置的隐状态
        # 创建一个范围张量 [0, 1, ..., seq_len-1]
        seq_lengths = attention_mask.sum(dim=1) - 1  # [batch_size//2]
        batch_indices = torch.arange(half_batch_size, device=input_ids.device)
        context_embedding = last_hidden_states[batch_indices, seq_lengths]  # [batch_size//2, hidden_size]
        
        # 计算动态 beta 值
        betas = self.beta_head.get_dynamic_beta(context_embedding, ppls)
        
        # 分割 chosen 和 rejected 样本的 logps 和 logits
        chosen_logps, rejected_logps = all_logps.split(half_batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(half_batch_size, dim=0)
        chosen_length, _ = valid_length.split(half_batch_size, dim=0)
        
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            chosen_logps = chosen_logps / chosen_length
            rejected_logps = rejected_logps / valid_length[half_batch_size:]
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps, betas, ppls
        else:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length, betas, ppls

    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        """计算批次的损失和指标
        
        Args:
            model: 策略模型
            batch: 输入数据，包含 chosen 和 rejected 样本
            train_eval: 指示是训练还是评估阶段
            
        Returns:
            losses.mean(): 平均损失值
            metrics: 包含各种指标的字典，包括每个输入的 beta 值
        """
        metrics = {}
        
        # 前向传播获取所有需要的值
        (
            policy_chosen_logps,  # [batch_size/2]
            policy_rejected_logps,  # [batch_size/2]
            policy_chosen_logits,  # [batch_size/2, seq_len, vocab_size]
            policy_rejected_logits,  # [batch_size/2, seq_len, vocab_size]
            policy_chosen_logps_avg,  # [batch_size/2]
            betas,  # [batch_size/2] 每个输入对应一个 beta 值
            ppls,   # [batch_size/2] 每个输入的困惑度值
        ) = self.concatenated_forward(model, batch)

        # 计算参考模型的 log probabilities
        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        
        # 计算损失和奖励
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            betas=betas  # 每个输入都有自己的 beta 值
        )

        # 计算 SFT loss
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        # 记录指标
        prefix = "eval_" if train_eval == "eval" else ""
        
        metrics.update({
            # 1. 基础指标：直接使用 mean() 计算平均值
            f"{prefix}rewards/chosen": chosen_rewards.mean().item(),
            f"{prefix}rewards/rejected": rejected_rewards.mean().item(),
            f"{prefix}rewards/margins": (chosen_rewards - rejected_rewards).mean().item(),
            f"{prefix}rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean().item(),
            
            # 2. 模型输出指标
            f"{prefix}logps/chosen": policy_chosen_logps.mean().item(),
            f"{prefix}logps/rejected": policy_rejected_logps.mean().item(),
            f"{prefix}logits/chosen": policy_chosen_logits.mean().item(),
            f"{prefix}logits/rejected": policy_rejected_logits.mean().item(),
            
            # 3. beta 和 ppl 指标：只需要平均值
            f"{prefix}beta": betas.mean().item(),
            f"{prefix}ppl": ppls.mean().item(),
        })

        if self.loss_type == "orpo":
            metrics.update({
                f"{prefix}sft_loss": sft_loss.mean().item(),
                f"{prefix}odds_ratio_loss": ((losses - sft_loss) / betas).mean().item()
            })

        return losses.mean(), metrics

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
        betas: "torch.Tensor"  # [batch_size/2] 每个输入对应一个 beta 值
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """计算偏好学习的损失
        
        每个输入文本都有自己的 beta 值，用于计算对应的 chosen 和 rejected 样本的损失
        """
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    betas  # 每个样本对使用对应的 beta 值
                )
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    betas  # 每个样本对使用对应的 beta 值
                )
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

            chosen_rewards = betas * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = betas * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            # 标准 DPO loss
            policy_logratios = policy_chosen_logps - policy_rejected_logps
            if reference_chosen_logps is not None and reference_rejected_logps is not None:
                reference_logratios = reference_chosen_logps - reference_rejected_logps
                logits = policy_logratios - reference_logratios
            else:
                logits = policy_logratios

            if self.label_smoothing > 0:
                losses = (
                    -F.logsigmoid(betas * logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-betas * logits) * self.label_smoothing
                )
            else:
                losses = -F.logsigmoid(betas * logits)

            chosen_rewards = policy_chosen_logps
            rejected_rewards = policy_rejected_logps

        return losses, chosen_rewards, rejected_rewards

    @override
    def compute_reference_log_probs(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """计算参考模型的 log probabilities
        
        注意：参考模型的计算不需要梯度，所以这里使用 no_grad
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
            if self.is_deepspeed_enabled:
                ref_model.eval()
            else:
                # 使用 accelerator.is_local_main_process 来检查是否是分布式训练
                context = ref_model.no_sync() if hasattr(ref_model, "no_sync") and not self.accelerator.is_local_main_process else nullcontext()
                with context:
                    outputs = ref_model(**batch, return_dict=True, use_cache=False)
                    
            ref_logits = outputs.logits.to(torch.float32)
            ref_logps, valid_length = get_batch_logps(logits=ref_logits, labels=batch["labels"])

            if self.loss_type in ["ipo", "orpo", "simpo"]:
                ref_logps = ref_logps / valid_length

            batch_size = batch["input_ids"].size(0) // 2
            chosen_ref_logps, rejected_ref_logps = ref_logps.split(batch_size, dim=0)
            return chosen_ref_logps, rejected_ref_logps

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
