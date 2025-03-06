# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/dpo.py
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

# =====================================================================
# LEDPO (Learnable Beta DPO) 工作流模块
# =====================================================================
# 本模块实现了LEDPO (Learnable Beta Direct Preference Optimization)训练流程
# LEDPO是对标准DPO的扩展，其中beta参数（控制偏好学习强度的超参数）设计为可学习的参数
# 这使模型能够动态调整偏好学习的强度，可能带来更好的训练效果和收敛性能
# =====================================================================

from typing import TYPE_CHECKING, List, Optional

from ...data import PairwiseDataCollatorWithPadding, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import LEDPOTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments


def run_ledpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    """
    执行LEDPO (Learnable Beta DPO)训练流程的主函数
    
    LEDPO是标准DPO的扩展版本，其核心创新点在于将beta参数（控制偏好信号强度的超参数）
    设计为可学习的参数。这使模型能够在训练过程中自适应地调整偏好学习的强度。
    
    参数:
        model_args: 包含模型配置的参数对象
        data_args: 包含数据集配置的参数对象
        training_args: 包含训练配置的参数对象
        finetuning_args: 包含微调策略的参数对象
        callbacks: 可选的训练回调函数列表
    """
    # 加载分词器和预处理相关组件
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    # 获取模板并修复分词器，确保分词器与模板正确配合
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 加载和预处理训练数据集
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    
    # 加载预训练模型
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # 创建配对数据整理器，用于处理偏好数据（每对包含一个较好回答和一个较差回答）
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model,
        pad_to_multiple_of=8,  # 填充到8的倍数以提高计算效率
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )

    # 创建参考模型（reference model）
    # 在DPO类训练中，参考模型用于计算KL散度，防止策略模型偏离太远
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):  # 如果不提供参考模型且不是训练模式，使用模型自身作为参考
            ref_model = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        ref_model = None

    # 更新训练参数
    training_args.remove_unused_columns = False  # 对于多模态和配对数据集很重要

    # 初始化LEDPO训练器
    # 这里是LEDPO的核心实现，在trainer.py中定义了如何处理可学习beta参数
    trainer = LEDPOTrainer(
        model=model,              # 策略模型，将被优化
        ref_model=ref_model,      # 参考模型，用于计算KL散度
        args=training_args,       # 训练参数
        finetuning_args=finetuning_args,  # 微调参数，包含beta初始值等设置
        data_collator=data_collator,  # 数据整理器
        callbacks=callbacks,      # 训练回调
        **dataset_module,         # 数据集相关组件
        **tokenizer_module,       # 分词器相关组件
    )

    # 训练流程
    if training_args.do_train:
        # 执行模型训练
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()  # 保存训练后的模型
        
        # 如果需要，计算有效的token处理速度（用于性能评估）
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="rm"
            )

        # 记录训练指标
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
        # 如果需要且当前是主进程，绘制损失曲线图
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])

    # 评估流程
    if training_args.do_eval:
        # 执行模型评估
        metrics = trainer.evaluate(metric_key_prefix="eval")
        
        # 如果参考模型与策略模型相同，则无法计算奖励指标
        if id(model) == id(ref_model):  # 无法计算奖励指标（如果参考模型和主模型相同）
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
                
        # 记录评估指标
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 创建模型卡并推送到Hub（如果配置允许）
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
