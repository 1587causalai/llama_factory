import os
import sys
import yaml
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Literal
import datasets
from datasets import load_dataset

# 添加导入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llamafactory.hparams import (
    ModelArguments,
    DataArguments,
    FinetuningArguments
)
from src.llamafactory.data.loader import get_dataset
from src.llamafactory.data.data_utils import split_dataset

def test_betadpo_step1_config():
    """第一步：测试配置加载"""
    try:
        print("=== 步骤 1: 测试配置加载 ===")
        
        # 1. 加载配置
        parser = HfArgumentParser((
            ModelArguments,
            DataArguments,
            Seq2SeqTrainingArguments,
            FinetuningArguments
        ))
        
        config_file = os.path.join(os.path.dirname(__file__), "configs", "test_betadpo.yaml")
        print(f"\n正在加载配置文件: {config_file}")
        model_args, data_args, training_args, finetuning_args = parser.parse_yaml_file(config_file)
        
        # 2. 打印关键配置
        print(f"\n训练配置:")
        print(f"- Batch size: {training_args.per_device_train_batch_size}")
        print(f"- Gradient accumulation: {training_args.gradient_accumulation_steps}")
        print(f"- Learning rate: {training_args.learning_rate}")
        print(f"- Max samples: {data_args.max_samples}")
        print(f"- Logging steps: {training_args.logging_steps}")
        
        print("\n配置加载测试完成!")
        return model_args, data_args, training_args, finetuning_args
        
    except Exception as e:
        print("\n配置加载测试过程中发生错误:")
        print(str(e))
        return None

def test_betadpo_step2_model(model_args: ModelArguments):
    """第二步：测试模型加载"""
    try:
        print("\n=== 步骤 2: 测试模型加载 ===")
        
        # 1. 加载分词器
        print(f"\n正在加载分词器: {model_args.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("分词器加载成功!")
        
        # 2. 加载模型
        print(f"\n正在加载模型: {model_args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype="auto"
        )
        print("模型加载成功!")
        
        # 3. 简单的模型测试
        print("\n执行简单的模型测试...")
        test_input = "你好，请问你是谁？"
        inputs = tokenizer(test_input, return_tensors="pt")
        print(f"测试输入: {test_input}")
        
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.95,
            temperature=0.1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"模型输出: {response}")
        
        print("\n模型加载测试完成!")
        return model, tokenizer
        
    except Exception as e:
        print("\n模型加载测试过程中发生错误:")
        print(str(e))
        return None

def test_betadpo_step3_dataset(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: FinetuningArguments
):
    """第三步：测试数据集加载"""
    try:
        print("\n=== 步骤 3: 测试数据集加载 ===")
        
        # 1. 导入 Beta DPO 工作流
        from src.llamafactory.train.betadpo.workflow import run_betadpo
        
        # 2. 打印数据集配置
        print("\n数据集配置:")
        print(f"- 数据集路径: {data_args.dataset}")
        print(f"- 最大样本数: {data_args.max_samples}")
        print(f"- 模板: {data_args.template}")
        
        # 3. 暂时关闭训练&评估模式
        original_do_train = training_args.do_train
        original_do_eval = training_args.do_eval
        training_args.do_train = False
        training_args.do_eval = False
        
        # 4. 运行工作流进行数据加载
        print("\n开始加载数据集...")
        run_betadpo(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args
        )
        
        # 5. 恢复训练模式设置
        training_args.do_train = original_do_train
        training_args.do_eval = original_do_eval
        
        print("\n数据集加载测试完成!")
        return True
        
    except Exception as e:
        print("\n数据集加载测试过程中发生错误:")
        print(str(e))
        return False

def main():
    """主测试函数"""
    print("开始 Beta DPO 渐进式测试...")
    
    # 步骤 1: 配置加载
    config_result = test_betadpo_step1_config()
    if config_result is None:
        print("配置加载测试失败，终止后续测试")
        return
        
    model_args, data_args, training_args, finetuning_args = config_result
    print("\n第一阶段测试完成!")
    
    # 步骤 2: 模型加载
    model_result = test_betadpo_step2_model(model_args)
    if model_result is None:
        print("模型加载测试失败，终止后续测试")
        return
        
    model, tokenizer = model_result
    print("\n第二阶段测试完成!")
    
    # 步骤 3: 数据集加载
    dataset_result = test_betadpo_step3_dataset(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args
    )
    if not dataset_result:
        print("数据集加载测试失败，终止后续测试")
        return
        
    print("\n第三阶段测试完成!")

if __name__ == "__main__":
    main() 