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

def test_betadpo():
    try:
        print("Testing Beta DPO with test config...")
        
        # 1. 加载配置
        parser = HfArgumentParser((
            ModelArguments,
            DataArguments,
            Seq2SeqTrainingArguments,
            FinetuningArguments
        ))
        
        config_file = os.path.join(os.path.dirname(__file__), "configs", "test_betadpo.yaml")
        model_args, data_args, training_args, finetuning_args = parser.parse_yaml_file(config_file)
        
        # 2. 运行 Beta DPO
        print("\nRunning Beta DPO training...")
        from src.llamafactory.train.betadpo.workflow import run_betadpo
        
        # 打印一些关键配置
        print(f"\nTraining config:")
        print(f"- Batch size: {training_args.per_device_train_batch_size}")
        print(f"- Gradient accumulation: {training_args.gradient_accumulation_steps}")
        print(f"- Learning rate: {training_args.learning_rate}")
        print(f"- Max samples: {data_args.max_samples}")
        print(f"- Logging steps: {training_args.logging_steps}")
        
        # 3. 运行训练
        run_betadpo(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args
        )
        
        # 4. 检查输出
        # 检查是否生成了损失图
        loss_png = os.path.join(training_args.output_dir, "loss.png")
        if os.path.exists(loss_png):
            print(f"\nLoss plot saved to: {loss_png}")
        else:
            print("\nWarning: Loss plot was not generated!")
            
        # 检查是否保存了 beta 值的统计信息
        beta_stats = os.path.join(training_args.output_dir, "beta_stats.json")
        if os.path.exists(beta_stats):
            print(f"Beta statistics saved to: {beta_stats}")
        else:
            print("Warning: Beta statistics were not generated!")
        
        print("\nBeta DPO training completed!")
    except Exception as e:
        import traceback
        print("\nError occurred during Beta DPO testing:")
        print(traceback.format_exc())
        raise e

if __name__ == "__main__":
    test_betadpo() 