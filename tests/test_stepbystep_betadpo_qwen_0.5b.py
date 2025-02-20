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

def test_betadpo_qwen_step1_config():
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
        
        config_file = os.path.join(os.path.dirname(__file__), "configs", "test_betadpo_qwen_0.5b.yaml")
        print(f"\n正在加载配置文件: {config_file}")
        model_args, data_args, training_args, finetuning_args = parser.parse_yaml_file(config_file)
        
        # 2. 打印关键配置
        print(f"\n训练配置:")
        print(f"- 模型路径: {model_args.model_name_or_path}")
        print(f"- Batch size: {training_args.per_device_train_batch_size}")
        print(f"- 梯度累积步数: {training_args.gradient_accumulation_steps}")
        print(f"- 学习率: {training_args.learning_rate}")
        print(f"- 最大样本数: {data_args.max_samples}")
        print(f"- 日志记录步数: {training_args.logging_steps}")
        print(f"- LoRA rank: {finetuning_args.lora_rank}")
        print(f"- 输出目录: {training_args.output_dir}")
        print(f"- Beta Head 类型: {finetuning_args.beta_head_type}")
        print(f"- Beta Head Epsilon: {finetuning_args.beta_head_epsilon}")
        
        print("\n配置加载测试完成!")
        return model_args, data_args, training_args, finetuning_args
        
    except Exception as e:
        print("\n配置加载测试过程中发生错误:")
        print(str(e))
        return None

def test_betadpo_qwen_step2_model(model_args: ModelArguments):
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

def test_betadpo_qwen_step3_dataset(
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

def test_betadpo_qwen_step4_training(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: FinetuningArguments
):
    """第四步：测试训练流程"""
    try:
        print("\n=== 步骤 4: 测试训练流程 ===")
        
        # 1. 设置训练模式（测试阶段不需要评估）
        training_args.do_train = True
        training_args.do_eval = False  # 关闭评估模式，因为测试时没有评估数据集
        
        # 2. 打印详细的训练配置
        print("\n训练配置详情:")
        print(f"- 模型路径: {model_args.model_name_or_path}")
        print(f"- 训练阶段: {finetuning_args.stage}")
        print(f"- 微调方法: {finetuning_args.finetuning_type}")
        print(f"- LoRA 配置:")
        print(f"  - Rank: {finetuning_args.lora_rank}")
        print(f"  - Alpha: {finetuning_args.lora_alpha}")
        print(f"  - Target: {finetuning_args.lora_target}")
        print(f"- Beta DPO 配置:")
        print(f"  - Beta Head Type: {finetuning_args.beta_head_type}")
        print(f"  - Beta Head Epsilon: {finetuning_args.beta_head_epsilon}")
        print(f"  - Pref Loss: {finetuning_args.pref_loss}")
        print(f"  - Pref Beta: {finetuning_args.pref_beta}")
        print(f"- 数据集配置:")
        print(f"  - 数据集: {data_args.dataset}")
        print(f"  - 最大样本数: {data_args.max_samples}")
        print(f"  - 模板: {data_args.template}")
        print(f"- 训练参数:")
        print(f"  - Batch Size: {training_args.per_device_train_batch_size}")
        print(f"  - 梯度累积: {training_args.gradient_accumulation_steps}")
        print(f"  - 学习率: {training_args.learning_rate}")
        print(f"  - 训练轮数: {training_args.num_train_epochs}")
        
        # 3. 运行 Beta DPO 训练
        print("\n开始 Beta DPO 训练...")
        from src.llamafactory.train.betadpo.workflow import run_betadpo
        
        try:
            run_betadpo(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args
            )
        except Exception as train_error:
            print("\n训练过程中发生错误:")
            import traceback
            print(traceback.format_exc())
            raise train_error
        
        # 4. 检查训练输出
        loss_png = os.path.join(training_args.output_dir, "loss.png")
        if os.path.exists(loss_png):
            print(f"\nLoss 图表已保存到: {loss_png}")
        else:
            print("\n警告: Loss 图表未生成!")
        
        adapter_path = os.path.join(training_args.output_dir, "adapter_model")
        if os.path.exists(adapter_path):
            print(f"LoRA adapter 已保存到: {adapter_path}")
        else:
            print("警告: LoRA adapter 未保存!")
        
        beta_stats = os.path.join(training_args.output_dir, "beta_stats.json")
        if os.path.exists(beta_stats):
            print(f"Beta 统计信息已保存到: {beta_stats}")
        else:
            print("警告: Beta 统计信息未保存!")
        
        print("\n训练流程测试完成!")
        return True
        
    except Exception as e:
        print("\n训练流程测试过程中发生错误:")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """主测试函数"""
    print("开始 Qwen1.5-0.5B-Chat Beta DPO 渐进式测试...")
    
    # 步骤 1: 配置加载
    config_result = test_betadpo_qwen_step1_config()
    if config_result is None:
        print("配置加载测试失败，终止后续测试")
        return
        
    model_args, data_args, training_args, finetuning_args = config_result
    print("\n第一阶段测试完成!")
    
    # 步骤 2: 模型加载
    model_result = test_betadpo_qwen_step2_model(model_args)
    if model_result is None:
        print("模型加载测试失败，终止后续测试")
        return
        
    model, tokenizer = model_result
    print("\n第二阶段测试完成!")
    
    # 步骤 3: 数据集加载
    dataset_result = test_betadpo_qwen_step3_dataset(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args
    )
    if not dataset_result:
        print("数据集加载测试失败，终止后续测试")
        return
        
    print("\n第三阶段测试完成!")
    
    # 步骤 4: 训练流程
    training_result = test_betadpo_qwen_step4_training(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args
    )
    if not training_result:
        print("训练流程测试失败")
        return
        
    print("\n所有测试阶段已完成!")

if __name__ == "__main__":
    main() 