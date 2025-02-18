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
    print("Testing BetaDPO with test config...")
    
    # 1. 加载配置
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        Seq2SeqTrainingArguments,
        FinetuningArguments
    ))
    
    config_file = os.path.join(os.path.dirname(__file__), "configs", "test_betadpo.yaml")
    model_args, data_args, training_args, finetuning_args = parser.parse_yaml_file(config_file)
    
    # 2. 运行 BetaDPO
    print("\nRunning BetaDPO training...")
    from src.llamafactory.train.betadpo.workflow import run_betadpo
    
    run_betadpo(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args
    )
    print("\nBetaDPO training completed!")

if __name__ == "__main__":
    test_betadpo() 