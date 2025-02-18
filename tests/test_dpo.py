import os
import yaml
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from dataclasses import dataclass
from typing import Optional, Literal

# 添加导入路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.llamafactory.hparams import (
    ModelArguments,
    DataArguments,
    FinetuningArguments
)

def test_dpo():
    print("Testing DPO with test config...")
    
    # 1. 加载配置
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        Seq2SeqTrainingArguments,
        FinetuningArguments
    ))
    
    config_file = os.path.join(os.path.dirname(__file__), "configs", "test_dpo.yaml")
    model_args, data_args, training_args, finetuning_args = parser.parse_yaml_file(config_file)
    
    # 2. 运行 DPO
    print("\nRunning DPO training...")
    from src.llamafactory.train.dpo.workflow import run_dpo
    
    # 打印一些关键配置
    print(f"\nTraining config:")
    print(f"- Batch size: {training_args.per_device_train_batch_size}")
    print(f"- Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"- Learning rate: {training_args.learning_rate}")
    print(f"- Max samples: {data_args.max_samples}")
    print(f"- Logging steps: {training_args.logging_steps}")
    
    run_dpo(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args
    )
    
    # 检查是否生成了损失图
    loss_png = os.path.join(training_args.output_dir, "loss.png")
    if os.path.exists(loss_png):
        print(f"\nLoss plot saved to: {loss_png}")
    else:
        print("\nWarning: Loss plot was not generated!")
    
    print("\nDPO training completed!")

if __name__ == "__main__":
    test_dpo() 