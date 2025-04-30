# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
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

import math
import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Literal, Optional, Union, Tuple
from enum import Enum



import torch
import torch.nn.functional as F
from torch.distributions import Normal
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps, nested_detach
from .beta_head import HiddenStateBetaHead


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments

class FDivergenceType(Enum):
    REVERSE_KL = "reverse_kl"
    JS_DIVERGENCE = "js_divergence"
    ALPHA_DIVERGENCE = "alpha_divergence"

class FDivergenceConstants:
    ALPHA_DIVERGENCE_COEF_KEY = "alpha_divergence_coef"
    ALPHA_DIVERGENCE_COEF_DEFAULT = 1.0


# 这两个函数看起来“奇怪”的原因主要是它们的实现方式比较特殊，涉及到一些底层的数值计算细节。
# 然而，它们的设计是有实际意义的，主要是为了在数值计算中避免溢出问题。在实际使用中，这些函数可以帮助用户更安全地进行指数运算，特别是在处理大规模数据或复杂模型时。
def get_exp_cap(value, decimal=4):
    """
    Get the exponent cap of a value. This is used to cap the exponent of a value to avoid overflow.
    The formula is : log(value.dtype.max)
    E.g.
      For float32 data type, the maximum exponent value is 88.7228 to 4 decimal points.
    ```
    Args:
        value (`torch.Tensor`):
            The input tensor to obtain the data type
        decimal (`int`):
            The number of decimal points of the output exponent cap.
            eg: direct calling exp(log(torch.float32.max)) will result in inf
            so we cap the exponent to 88.7228 to avoid overflow.
    """
    vdtype_max = torch.zeros([1]).to(value.dtype) + torch.finfo(value.dtype).max
    vdtype_log_max = torch.log(vdtype_max).to(value.device)
    return torch.floor(vdtype_log_max * 10**decimal) / 10**decimal if decimal > 0 else vdtype_log_max


def cap_exp(value, cap=-1):
    # Cap the exponent value below the upper-bound to avoid overflow, before calling torch.exp
    cap = get_exp_cap(value) if cap < 0 else cap
    return torch.exp(torch.clamp(value, max=cap))


def compute_pref_probs(logits: torch.FloatTensor, beta: torch.FloatTensor | float, disco_pref: bool = False) -> torch.FloatTensor:
    """
    Compute the preference probabilities for a given logits and beta.  (后续计算数值稳定性考虑, 不用于计算 loss)

    Args:
        logits: Tensor of log probability ratios or reward differences. Shape: (batch_size,)
        beta: Temperature parameter for scaling the logits. Can be a scalar (float) or a tensor with the same shape as logits.
        disco_pref: If True, use disco-DPO preference probability with normal distribution assumption;
                   if False, use standard DPO sigmoid preference probability. (default: False)

    Returns:
        Tensor of preference probabilities. Shape: (batch_size,)
    """
    # 确保 beta 和 logits 的维度兼容
    if isinstance(beta, torch.Tensor) and beta.shape != logits.shape:
        raise ValueError(f"beta tensor shape {beta.shape} must match logits shape {logits.shape}")

    if disco_pref:
        # disco-DPO: p(y_w > y_l) = 1/2 * (1 + erf((β * logits)/√2))
        scaled_logits = beta * logits  # 广播机制自动处理标量或张量
        pref_probs = 0.5 * (1 + torch.erf(0.6 * scaled_logits / math.sqrt(2))) # 很容易数值溢出, 所以乘以 0.01
    else:
        # Standard DPO: p(y_w > y_l) = sigmoid(β * logits)
        pref_probs = torch.sigmoid(beta * logits)  # 广播机制自动处理标量或张量
    
    return pref_probs


def compute_log_pref_prob_customized(logits: torch.FloatTensor, beta: torch.FloatTensor | float, disco_pref: bool = False) -> torch.FloatTensor:
    """
    Compute the log preference probabilities for given logits and beta, avoiding numerical overflow.

    Args:
        logits: Tensor of log probability ratios or reward differences. Shape: (batch_size,)
        beta: Temperature parameter for scaling the logits. Can be a scalar (float) or a tensor with the same shape as logits.
        disco_pref: If True, use disco-DPO log preference probability with normal distribution assumption;
                   if False, use standard DPO log sigmoid preference probability. (default: False)


        要防止数值溢出，可以通过数学上的渐近展开，用解析近似替代直接计算。这种方法确保在极值区域内，结果保持有限且稳定，避免超出计算机的数值范围。更多内容请参考 https://grok.com/share/bGVnYWN5_7bfa15cf-9ed8-4be3-99e1-453d3f08305c

        当 \(x\) 很负时，\(\Phi(x)\)（正态分布 CDF）会变得极小，直接算 \(\log(\Phi(x))\) 容易下溢为 \(-inf\)。为避免溢出，可以用近似公式：

        \[
        \log(\Phi(x)) \approx -\frac{x^2}{2} - \frac{1}{2} \log(2\pi) - \log(-x)
        \]

        这个公式通过数学推导，把极小值转化为几个有限项的组合，避免直接计算 \(\Phi(x)\)。核心是：**用稳定的数学近似绕过浮点数限制**，既简单又有效。

    Returns:
        Tensor of log preference probabilities. Shape: (batch_size,)
    """
    # 确保 beta 和 logits 的维度兼容
    if isinstance(beta, torch.Tensor) and beta.shape != logits.shape:
        raise ValueError(f"beta tensor shape {beta.shape} must match logits shape {logits.shape}")

    if disco_pref:
        # disco-DPO: log(p(y_w > y_l)) = log(Φ(0.6 * β * logits / √2))
        scaled_logits = 0.6 * beta * logits / math.sqrt(2)
        return torch.special.log_ndtr(scaled_logits)  
    else:
        # Standard DPO: log(p(y_w > y_l)) = logsigmoid(β * logits)
        return torch.nn.functional.logsigmoid(beta * logits)


class CustomDPOTrainer(DPOTrainer):
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

        # customized params for dpo variants
        self.use_dynamic_beta = finetuning_args.use_dynamic_beta
        self.disco_pref = finetuning_args.disco_pref

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
            self.beta_head = HiddenStateBetaHead(
                hidden_size=model.config.hidden_size,
                beta_base=self.beta
            ).to(self.accelerator.device)
            self.beta_pos_delta = []
            self.beta_neg_delta = []

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            # 创建基本优化器
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)

            if self.optimizer is None:
                self.optimizer = super().create_optimizer()
            
            # 添加beta_head参数到优化器
            if self.use_dynamic_beta and hasattr(self, "beta_head"):

                
                beta_head_params = list(self.beta_head.parameters())
                
                print(f"[DEBUG] BetaHead parameters:")
                for name, param in self.beta_head.named_parameters():
                    print(f"[DEBUG]   {name}: shape={param.shape}, requires_grad={param.requires_grad}")
                
                # 为beta_head参数设置更高的学习率（例如，是基本学习率的10倍）
                # beta_head_lr = self.args.learning_rate * 10.0  # 10倍
                beta_head_lr = self.args.learning_rate * 0.3 # 慢一定更新, 不然会不小到了 beta -> 0 的局部最优解了. 
                # 添加参数组
                params_config = {
                    "params": beta_head_params,
                    "lr": beta_head_lr,  # 使用不同的学习率
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
    def get_batch_samples(self, *args, **kwargs):
        r"""Replace the method of DPO Trainer with the one of the standard Trainer."""
        return Trainer.get_batch_samples(self, *args, **kwargs)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor", beta_values: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        r"""Compute ORPO's odds ratio (OR) loss for batched log probabilities of the policy model."""
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -torch.log(compute_pref_probs(log_odds, 1.0, self.disco_pref))  # 使用1.0作为beta值, 后面的 beta_values 是动态的
        beta = beta_values if beta_values is not None else self.beta
        orpo_loss = sft_loss + beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor", beta_values: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        r"""Compute SimPO loss for batched log probabilities of the policy model."""
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        beta = beta_values if beta_values is not None else self.beta
        simpo_loss = -torch.log(compute_pref_probs(logits, beta, self.disco_pref))
        return simpo_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
        beta_values: Optional["torch.Tensor"] = None
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""Compute loss for preference learning."""
        delta = (policy_chosen_logps - reference_chosen_logps) - (policy_rejected_logps - reference_rejected_logps)

        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps, beta_values)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps, beta_values)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")
            
            beta = beta_values if beta_values is not None else self.beta
            chosen_rewards = beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta_values
            )

        delta_pos_mask = delta > 0
        delta_neg_mask = delta <= 0

        if delta_neg_mask.sum() > 0:
            debug_loss = delta_neg_mask * losses
        else:
            debug_loss = delta_pos_mask * losses


        return losses, chosen_rewards, rejected_rewards

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""Compute the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # avoid error
        if self.use_dynamic_beta:
            model_outputs = model(**batch, return_dict=True, use_cache=False, output_hidden_states=True)
            all_logits: torch.Tensor = model_outputs.logits.to(torch.float32)
            last_layer_hidden_states: torch.Tensor = model_outputs.hidden_states[-1].to(torch.float32) # [batch_size, seq_len, hidden_size]

            all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
            if self.loss_type in ["ipo", "orpo", "simpo"]:
                all_logps = all_logps / valid_length

            batch_size = batch["input_ids"].size(0) // 2
            chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
            chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
            chosen_length, _ = valid_length.split(batch_size, dim=0)

            chosen_hidden_states, rejected_hidden_states = last_layer_hidden_states.split(batch_size, dim=0)
            prompt_lengths = self.get_prompt_lengths(batch)  
            prompt_idx = (prompt_lengths - 1).clamp(0, chosen_hidden_states.size(1) - 1)
            batch_idx = torch.arange(batch_size)
            prompt_last_token_hidden_states = chosen_hidden_states[batch_idx, prompt_idx] # [batch_size, hidden_size]
            beta_values = self.beta_head(prompt_last_token_hidden_states)

            if self.loss_type in ["ipo", "orpo", "simpo"]:
                return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps, beta_values
            else:
                return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length, beta_values
            
        else:
            
            all_logits: torch.Tensor = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
            all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
            if self.loss_type in ["ipo", "orpo", "simpo"]:
                all_logps = all_logps / valid_length

            batch_size = batch["input_ids"].size(0) // 2
            chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
            chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
            chosen_length, _ = valid_length.split(batch_size, dim=0)
            beta_values = None

            if self.loss_type in ["ipo", "orpo", "simpo"]:
                return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps, beta_values
            else:
                return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length, beta_values

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
    

    def ref_model_forward(self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute log probabilities of the reference model with beta head."""

        if not self.finetuning_args.use_ref_model:
            return None, None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with ref_context:
            # 调用 concatenated_forward 方法
            outputs = self.concatenated_forward(ref_model, batch)
        
        # reference_chosen_logps, reference_rejected_logps 计算, 并且 detach
        reference_chosen_logps, reference_rejected_logps, *_ = outputs
        reference_chosen_logps = reference_chosen_logps.detach()
        reference_rejected_logps = reference_rejected_logps.detach()

        # 获取 beta_values
        beta_values = outputs[-1]

        return reference_chosen_logps, reference_rejected_logps, beta_values


    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        r"""Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

    
        policy_outputs = self.concatenated_forward(model, batch)
        policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits, policy_chosen_logps_avg = policy_outputs[:5]

        reference_chosen_logps, reference_rejected_logps, beta_values = self.ref_model_forward(model, batch)

        beta = beta_values if self.use_dynamic_beta else self.beta

        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta_values
        )

        # print(f"losses: {losses}")
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
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / beta).mean().item()
        
        metrics[f"{prefix}beta"] = beta_values.mean().item() if self.use_dynamic_beta else self.beta
        delta = (policy_chosen_logps - reference_chosen_logps) - (policy_rejected_logps - reference_rejected_logps)
        metrics[f"{prefix}delta"] = delta.mean().item()

        if self.use_dynamic_beta:
            beta_pos_delta = beta_values[delta > 0]
            beta_neg_delta = beta_values[delta <= 0]
            # 一个batch 太小, 很容易为空, 我也不知道填充一个什么样的值比较好, 所以就填充一个平均值.
            # fill_value = beta_values.mean().item()
            # metrics[f"{prefix}beta_pos_delta"] = pos_delta_beta.mean().item() if pos_delta_beta.numel() > 0 else fill_value
            # metrics[f"{prefix}beta_neg_delta"] = neg_delta_beta.mean().item() if neg_delta_beta.numel() > 0 else fill_value
            if beta_pos_delta.numel() > 0:
                self.beta_pos_delta.append(beta_pos_delta.mean().item())
            if beta_neg_delta.numel() > 0:
                self.beta_neg_delta.append(beta_neg_delta.mean().item())

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


    @override
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        beta_values: Optional["torch.Tensor"] = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        chosen_logratios = policy_chosen_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_chosen_logps.to(self.accelerator.device)
        rejected_logratios = policy_rejected_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_rejected_logps.to(self.accelerator.device)

        beta = beta_values if beta_values is not None else self.beta

        def log_pref_prob(logits, beta=1.0):
            """
            计算偏好概率的对数

            beta 是偏好概率的参数, 默认是1.0, 这里有个非常好的技巧 log_pref_prob(logits, beta) 和 log_pref_prob(logits * beta, 1.0) 是等价的.
            """
            if self.disco_pref:
                return compute_log_pref_prob_customized(logits, beta, True)
            else:
                return compute_log_pref_prob_customized(logits, beta, False)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
            else:
                ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios = pi_logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = pi_logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)
        if self.loss_type == "sigmoid":
            losses = (
                -log_pref_prob(logits, beta) * (1 - self.label_smoothing)
                - log_pref_prob(-logits, beta) * self.label_smoothing
            )
        elif self.loss_type == "robust":
            losses = (
                -log_pref_prob(logits, beta) * (1 - self.label_smoothing)
                + log_pref_prob(-logits, beta) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == "exo_pair":
            import math

            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (beta * logits).sigmoid() * (
                log_pref_prob(logits, beta) - math.log(1 - self.label_smoothing)
            ) + (-beta * logits).sigmoid() * (log_pref_prob(-logits, beta) - math.log(self.label_smoothing))
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - beta * logits)
        elif self.loss_type == "ipo":
            losses = (logits - 1 / (2 * beta)) ** 2
        elif self.loss_type == "bco_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_rewards = beta * chosen_logratios
            rejected_rewards = beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean

            losses = -log_pref_prob((beta * chosen_logratios) - delta) - log_pref_prob(-(beta * rejected_logratios - delta))
        elif self.loss_type == "sppo_hard":
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            losses = (a - 0.5 / beta) ** 2 + (b + 0.5 / beta) ** 2
        elif self.loss_type == "nca_pair":
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * beta
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * beta
            losses = (
                -log_pref_prob(chosen_rewards)
                - 0.5 * log_pref_prob(-chosen_rewards)
                - 0.5 * log_pref_prob(-rejected_rewards)
            )
        elif self.loss_type == "aot_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)

            delta = chosen_logratios_sorted - rejected_logratios_sorted

            losses = (
                -log_pref_prob(beta * delta) * (1 - self.label_smoothing)
                - log_pref_prob(-beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "aot":
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios_sorted, _ = torch.sort(pi_logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)

            delta = pi_logratios_sorted - ref_logratios_sorted

            losses = (
                -log_pref_prob(beta * delta) * (1 - self.label_smoothing)
                - log_pref_prob(-beta * delta) * self.label_smoothing
            )

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'bco_pair', 'sppo_hard', 'nca_pair', 'robust', 'exo_pair']"
            )

        chosen_rewards = (
            beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards


    def get_prompt_lengths(self, batch: dict[str, "torch.Tensor"]) -> torch.Tensor:
        """
        计算batch中每个样本的prompt长度
        
        在DPO/LEDPO训练中，batch包含成对的样本(chosen和rejected)
        标签中IGNORE_INDEX表示prompt部分
        
        返回: [batch_size//2] 形状的张量，只包含chosen样本的prompt长度
        """
        # DPO batch中结构: [chosen_1, ..., chosen_n, rejected_1, ..., rejected_n]
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