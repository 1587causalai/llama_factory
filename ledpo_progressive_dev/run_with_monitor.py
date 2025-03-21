#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接运行LEDPO训练并监控beta_head参数变化

使用方法:
    python ledpo_progressive_dev/run_with_monitor.py --config ledpo_progressive_dev/qwen15_lora_foodpo.yaml [--wandb_project PROJECT]
"""

import argparse
import os
import sys
import yaml
import torch
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from transformers.trainer_callback import TrainerCallback
import matplotlib.pyplot as plt


class BetaHeadMonitorCallback(TrainerCallback):
    """监控beta_head参数变化的回调"""
    
    def __init__(self, check_interval=100):
        self.initial_params = {}
        self.param_history = {}
        self.check_interval = check_interval
        self.save_dir = None
        self.gradient_history = {}
        self.trainer = None
        self.beta_head = None  # 新增：直接存储beta_head引用
    
    def set_beta_head(self, beta_head):
        """手动设置beta_head引用"""
        if beta_head is not None:
            self.beta_head = beta_head
            print("\n[BETA-MONITOR] beta_head已手动设置")
            return True
        return False
    
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时记录初始参数"""
        # 保存trainer的引用
        self.trainer = kwargs.get("trainer", None)
        
        # 首先检查是否已手动设置beta_head
        if self.beta_head is not None:
            beta_head = self.beta_head
        # 然后尝试直接访问beta_head
        elif self.trainer is not None and hasattr(self.trainer, "beta_head"):
            beta_head = self.trainer.beta_head
        # 如果没有，尝试从model属性访问
        elif self.trainer is not None and hasattr(self.trainer, "model") and hasattr(self.trainer.model, "beta_head"):
            beta_head = self.trainer.model.beta_head
        else:
            print("\n[BETA-MONITOR] 未找到beta_head，跳过监控...")
            return
        
        # 创建保存目录
        self.save_dir = os.path.join(args.output_dir, "beta_head_monitor")
        os.makedirs(self.save_dir, exist_ok=True)
            
        print("\n[BETA-MONITOR] 开始监控beta_head参数...")
        with torch.no_grad():
            for name, param in beta_head.named_parameters():
                # 记录初始参数
                param_data = param.detach().cpu()
                self.initial_params[name] = param_data.clone()
                
                # 初始化历史记录
                self.param_history[name] = []
                self.gradient_history[name] = []
                
                # 打印初始信息
                print(f"[BETA-MONITOR] 初始参数 {name}:")
                print(f"  - 形状: {param.shape}")
                print(f"  - 范数: {param_data.norm().item():.8f}")
                print(f"  - 参数需要梯度: {param.requires_grad}")
                
                # 记录部分数值
                flat_data = param_data.flatten()
                if len(flat_data) > 5:
                    print(f"  - 前5个值: {flat_data[:5].tolist()}")
                else:
                    print(f"  - 所有值: {flat_data.tolist()}")
        
        # 检查优化器中是否包含beta_head参数
        self._check_optimizer()
        
        # 记录到文件
        self._save_checkpoint(state, "initial")
    
    def on_step_end(self, args, state, control, **kwargs):
        """每隔一定步数检查参数变化"""
        if state.global_step % self.check_interval != 0 or self.trainer is None:
            return
        
        # 获取beta_head对象
        if self.beta_head is not None:
            beta_head = self.beta_head
        elif hasattr(self.trainer, "beta_head"):
            beta_head = self.trainer.beta_head
        elif hasattr(self.trainer, "model") and hasattr(self.trainer.model, "beta_head"):
            beta_head = self.trainer.model.beta_head
        else:
            return
            
        # 记录参数和梯度
        self._record_params_and_grads(beta_head, state)
        
        # 保存检查点
        self._save_checkpoint(state, f"step_{state.global_step}")
        
        # 打印当前状态
        self._print_current_state(beta_head, state)
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时比较参数变化"""
        if self.trainer is None:
            return
            
        # 获取beta_head对象
        if self.beta_head is not None:
            beta_head = self.beta_head
        elif hasattr(self.trainer, "beta_head"):
            beta_head = self.trainer.beta_head
        elif hasattr(self.trainer, "model") and hasattr(self.trainer.model, "beta_head"):
            beta_head = self.trainer.model.beta_head
        else:
            return
            
        # 保存最终状态
        self._save_checkpoint(state, "final")
        
        # 打印详细比较报告
        print("\n" + "="*60)
        print("[BETA-MONITOR] 训练结束，参数变化报告:")
        print("="*60)
        
        with torch.no_grad():
            for name, param in beta_head.named_parameters():
                if name in self.initial_params:
                    initial = self.initial_params[name]
                    current = param.detach().cpu()
                    
                    # 计算各种指标
                    initial_norm = initial.norm().item()
                    current_norm = current.norm().item()
                    abs_diff = (current - initial).norm().item()
                    rel_diff = (abs_diff / initial_norm * 100) if initial_norm > 0 else float('inf')
                    
                    print(f"\n参数: {name}")
                    print(f"  - 初始范数: {initial_norm:.8f}")
                    print(f"  - 最终范数: {current_norm:.8f}")
                    print(f"  - 绝对变化: {abs_diff:.8f}")
                    print(f"  - 相对变化: {rel_diff:.4f}%")
                    
                    # 比较部分实际值
                    initial_flat = initial.flatten()
                    current_flat = current.flatten()
                    if len(initial_flat) > 0:
                        sample_size = min(5, len(initial_flat))
                        print(f"\n  样本值比较 (前{sample_size}个):")
                        for i in range(sample_size):
                            print(f"    [{i}] 初始: {initial_flat[i].item():.8f} -> 最终: {current_flat[i].item():.8f} (变化: {(current_flat[i]-initial_flat[i]).item():.8f})")
        
        # 生成总结
        has_changed = any((param.detach().cpu() - self.initial_params[name]).norm().item() > 1e-6 
                         for name, param in beta_head.named_parameters() 
                         if name in self.initial_params)
        
        print("\n" + "="*60)
        if has_changed:
            print("[BETA-MONITOR] 结论: beta_head参数在训练过程中已更新!")
        else:
            print("[BETA-MONITOR] 结论: beta_head参数在训练过程中未发生明显变化!")
        print("="*60)
        
        # 保存完整历史记录
        self._save_history_report()
        
        # 生成历史图表
        self._generate_history_plots()
    
    def _check_optimizer(self):
        """检查优化器中是否包含beta_head参数"""
        if self.trainer is None or not hasattr(self.trainer, "optimizer") or self.trainer.optimizer is None:
            print("[BETA-MONITOR] 警告: 优化器尚未初始化，无法检查参数")
            return
            
        # 获取beta_head对象
        if self.beta_head is not None:
            beta_head = self.beta_head
        elif hasattr(self.trainer, "beta_head"):
            beta_head = self.trainer.beta_head
        elif hasattr(self.trainer, "model") and hasattr(self.trainer.model, "beta_head"):
            beta_head = self.trainer.model.beta_head
        else:
            print("[BETA-MONITOR] 警告: 未找到beta_head，无法检查优化器配置")
            return
            
        print("\n[BETA-MONITOR] 检查优化器配置...")
        
        # 获取beta_head参数ID列表
        beta_head_param_ids = {id(p) for n, p in beta_head.named_parameters() if p.requires_grad}
        
        # 检查优化器参数组
        found_in_optimizer = False
        for i, group in enumerate(self.trainer.optimizer.param_groups):
            params_in_group = len(group["params"])
            lr = group.get("lr", None)
            
            # 检查此组中有多少beta_head参数
            beta_params_in_group = sum(1 for p in group["params"] if id(p) in beta_head_param_ids)
            
            print(f"[BETA-MONITOR] 参数组 {i}: {params_in_group}个参数, 学习率={lr}")
            if beta_params_in_group > 0:
                found_in_optimizer = True
                print(f"[BETA-MONITOR] 参数组 {i} 包含 {beta_params_in_group} 个beta_head参数")
            else:
                print(f"[BETA-MONITOR] 参数组 {i} 不包含beta_head参数")
        
        if not found_in_optimizer:
            print("[BETA-MONITOR] 警告: beta_head参数未在优化器中找到! 这可能是问题所在。")
    
    def _record_params_and_grads(self, beta_head, state):
        """记录参数和梯度状态"""
        with torch.no_grad():
            for name, param in beta_head.named_parameters():
                if not param.requires_grad:
                    continue
                    
                # 记录参数
                param_data = param.detach().cpu()
                self.param_history[name].append({
                    "step": state.global_step,
                    "norm": param_data.norm().item(),
                    "mean": param_data.mean().item()
                })
                
                # 记录梯度，如果存在
                if param.grad is not None:
                    grad_data = param.grad.detach().cpu()
                    self.gradient_history[name].append({
                        "step": state.global_step,
                        "norm": grad_data.norm().item(),
                        "mean": grad_data.mean().item(),
                        "max": grad_data.abs().max().item()
                    })
                else:
                    self.gradient_history[name].append({
                        "step": state.global_step,
                        "norm": 0.0,
                        "mean": 0.0,
                        "max": 0.0
                    })
    
    def _save_checkpoint(self, state, checkpoint_name):
        """保存当前参数检查点"""
        if not self.save_dir or self.trainer is None:
            return
            
        # 获取beta_head对象
        if self.beta_head is not None:
            beta_head = self.beta_head
        elif hasattr(self.trainer, "beta_head"):
            beta_head = self.trainer.beta_head
        elif hasattr(self.trainer, "model") and hasattr(self.trainer.model, "beta_head"):
            beta_head = self.trainer.model.beta_head
        else:
            return
            
        checkpoint_data = {
            "step": state.global_step,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {}
        }
        
        with torch.no_grad():
            for name, param in beta_head.named_parameters():
                param_data = param.detach().cpu()
                
                # 为每个参数保存范数和有限样本
                param_info = {
                    "norm": param_data.norm().item(),
                    "mean": param_data.mean().item(),
                    "std": param_data.std().item(),
                    "requires_grad": param.requires_grad
                }
                
                # 如果参数不太大，保存所有值；否则只保存样本
                flat_data = param_data.flatten()
                if len(flat_data) <= 20:
                    param_info["values"] = flat_data.tolist()
                else:
                    param_info["samples"] = flat_data[:10].tolist()
                
                # 记录梯度信息，如果存在
                if param.grad is not None:
                    grad_data = param.grad.detach().cpu()
                    param_info["grad"] = {
                        "norm": grad_data.norm().item(),
                        "mean": grad_data.mean().item(),
                        "max": grad_data.abs().max().item()
                    }
                
                checkpoint_data["parameters"][name] = param_info
        
        # 保存到文件
        save_path = os.path.join(self.save_dir, f"beta_params_{checkpoint_name}.json")
        with open(save_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def _print_current_state(self, beta_head, state):
        """打印当前参数状态"""
        print(f"\n[BETA-MONITOR] 步骤 {state.global_step} 参数状态:")
        
        with torch.no_grad():
            for name, param in beta_head.named_parameters():
                if name in self.initial_params:
                    initial = self.initial_params[name]
                    current = param.detach().cpu()
                    
                    # 计算变化
                    abs_diff = (current - initial).norm().item()
                    rel_diff = (abs_diff / initial.norm().item() * 100) if initial.norm().item() > 0 else float('inf')
                    
                    # 输出状态
                    print(f"  {name}: 范数={current.norm().item():.6f}, 变化={abs_diff:.6f} ({rel_diff:.2f}%)")
                    
                    # 如果有梯度，报告梯度信息
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        print(f"    - 梯度范数: {grad_norm:.8f}")
                    else:
                        print(f"    - 梯度: None")
    
    def _save_history_report(self):
        """保存完整的参数历史报告"""
        if not self.save_dir:
            return
            
        # 生成历史报告
        history_data = {
            "parameters": self.param_history,
            "gradients": self.gradient_history
        }
        
        history_path = os.path.join(self.save_dir, "param_history.json")
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)
            
        print(f"[BETA-MONITOR] 参数历史记录已保存到: {history_path}")
    
    def _generate_history_plots(self):
        """生成参数变化图表"""
        if not self.save_dir:
            return
            
        try:
            # 为每个参数生成图表
            for name in self.param_history:
                self._plot_parameter_history(name)
                
            # 为每个梯度生成图表
            for name in self.gradient_history:
                self._plot_gradient_history(name)
                
        except ImportError:
            print("[BETA-MONITOR] 未安装matplotlib，跳过图表生成")
        except Exception as e:
            print(f"[BETA-MONITOR] 生成图表时出错: {e}")
    
    def _plot_parameter_history(self, param_name):
        """绘制参数历史记录图表"""
        history = self.param_history[param_name]
        if not history:
            return
            
        # 提取数据
        steps = [entry["step"] for entry in history]
        norms = [entry["norm"] for entry in history]
        means = [entry["mean"] for entry in history]
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 绘制范数
        plt.subplot(1, 2, 1)
        plt.plot(steps, norms, 'b-o', linewidth=2, markersize=4)
        plt.title(f"Parameter Norm: {param_name}")
        plt.xlabel("Training Steps")
        plt.ylabel("L2 Norm")
        plt.grid(True)
        
        # 绘制均值
        plt.subplot(1, 2, 2)
        plt.plot(steps, means, 'r-o', linewidth=2, markersize=4)
        plt.title(f"Parameter Mean: {param_name}")
        plt.xlabel("Training Steps")
        plt.ylabel("Mean Value")
        plt.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{param_name.replace('.', '_')}_param.png")
        plt.savefig(save_path)
        plt.close()
    
    def _plot_gradient_history(self, param_name):
        """绘制梯度历史记录图表"""
        history = self.gradient_history[param_name]
        if not history:
            return
            
        # 提取数据
        steps = [entry["step"] for entry in history]
        norms = [entry["norm"] for entry in history]
        maxs = [entry["max"] for entry in history]
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 绘制范数
        plt.subplot(1, 2, 1)
        plt.plot(steps, norms, 'g-o', linewidth=2, markersize=4)
        plt.title(f"Gradient Norm: {param_name}")
        plt.xlabel("Training Steps")
        plt.ylabel("L2 Norm")
        plt.grid(True)
        
        # 绘制最大值
        plt.subplot(1, 2, 2)
        plt.plot(steps, maxs, 'm-o', linewidth=2, markersize=4)
        plt.title(f"Gradient Max: {param_name}")
        plt.xlabel("Training Steps")
        plt.ylabel("Max Absolute Value")
        plt.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{param_name.replace('.', '_')}_grad.png")
        plt.savefig(save_path)
        plt.close()


def run_with_monitor(config_path, wandb_project=None):
    """运行训练并监控beta_head参数"""
    # 导入必要的模块
    from llamafactory.hparams import get_train_args, read_args
    from llamafactory.model import load_model, load_tokenizer
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, PairwiseDataCollatorWithPadding
    from llamafactory.train.foodpo.trainer import CustomFooDPOTrainer
    from llamafactory.train.trainer_utils import create_ref_model
    from llamafactory.train.callbacks import LogCallback, ReporterCallback
    from llamafactory.extras.constants import IGNORE_INDEX
    
    # 设置wandb项目（如果提供）
    if wandb_project:
        os.environ['WANDB_PROJECT'] = wandb_project
    
    print('=' * 60)
    print(f"正在加载配置: {config_path}")
    print('=' * 60)
    
    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 将配置转换为命令行参数格式
    args = []
    for key, value in config.items():
        if isinstance(value, bool):
            # 修改布尔值处理逻辑，确保False值也能被传递
            args.append(f"--{key}={'true' if value else 'false'}")
        elif isinstance(value, list):
            args.append(f"--{key}={','.join(map(str, value))}")
        elif value is not None:
            args.append(f"--{key}={value}")
    
    # 获取训练参数
    print('=' * 60)
    print("处理参数中...")
    print('=' * 60)
    
    # 使用正确的方式获取参数
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    
    # 直接从YAML设置关键参数 - 解决布尔值参数解析问题
    if 'use_dynamic_beta' in config:
        finetuning_args.use_dynamic_beta = config['use_dynamic_beta']
        print(f"直接设置 use_dynamic_beta = {finetuning_args.use_dynamic_beta}")
    
    # 确保输出目录存在
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # 确保remove_unused_columns=False - 这对于DPO训练非常重要
    training_args.remove_unused_columns = False
    print(f"强制设置 remove_unused_columns = {training_args.remove_unused_columns}")
    
    # 阶段1: 准备模型组件
    print('=' * 60)
    print("准备模型组件...")
    print('=' * 60)
    
    # 1. 加载tokenizer
    print("加载Tokenizer...")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    # 2. 获取模板
    print("获取模板...")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 3. 加载模型
    print(f"加载模型: {model_args.model_name_or_path}...")
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    
    # 4. 创建参考模型
    print("准备参考模型...")
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):
            print("参考模型与主模型相同")
            ref_model = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        print("未使用参考模型")
        ref_model = None
    
    # 阶段2: 准备数据集
    print('=' * 60)
    print("准备数据集...")
    print('=' * 60)
    
    # 获取数据集
    dataset_module = get_dataset(
        template, 
        model_args, 
        data_args, 
        training_args, 
        stage="rm",  # DPO使用RM阶段的数据处理逻辑
        **tokenizer_module
    )
    
    # 创建数据整理器
    print("创建数据整理器...")
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )
    
    # 打印数据集信息
    if "train_dataset" in dataset_module:
        train_size = len(dataset_module["train_dataset"])
        print(f"训练集样本数: {train_size}")
    
    if "eval_dataset" in dataset_module and dataset_module["eval_dataset"] is not None:
        eval_size = len(dataset_module["eval_dataset"])
        print(f"验证集样本数: {eval_size}")
    
    # 阶段3: 设置训练器
    print('=' * 60)
    print("设置训练器...")
    print('=' * 60)
    
    # 创建回调函数
    callbacks = []
    callbacks.append(LogCallback())
    
    # 添加beta监控回调
    beta_monitor = BetaHeadMonitorCallback()
    callbacks.append(beta_monitor)
    
    # 添加Reporter回调
    callbacks.append(ReporterCallback(
        model_args=model_args,
        data_args=data_args,
        finetuning_args=finetuning_args, 
        generating_args=generating_args
    ))
    
    # 初始化DPO训练器
    print("初始化FooDPO训练器...")
    trainer = CustomFooDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )
    
    print(f"训练设备: {training_args.device}")
    
    # 初始化完trainer后，尝试不同路径获取beta_head
    print("检查trainer是否有beta_head属性:", hasattr(trainer, "beta_head"))
    
    # 添加更多日志以帮助诊断beta_head位置
    print("\n[DEBUG] 尝试确定beta_head位置:")
    beta_head = None
    
    if hasattr(trainer, "beta_head"):
        print("[DEBUG] trainer.beta_head 存在")
        beta_head_location = "trainer.beta_head"
        beta_head = trainer.beta_head
    elif hasattr(trainer.model, "beta_head"):
        print("[DEBUG] trainer.model.beta_head 存在")
        beta_head_location = "trainer.model.beta_head"
        beta_head = trainer.model.beta_head
    else:
        # 递归搜索beta_head
        print("[DEBUG] 在trainer对象中递归搜索beta_head...")
        beta_head_location = None
        
        def find_beta_head(obj, path="trainer", max_depth=3):
            """递归搜索对象中的beta_head属性"""
            if max_depth <= 0:
                return None, None
                
            for attr_name in dir(obj):
                # 跳过私有属性和方法
                if attr_name.startswith('_') or callable(getattr(obj, attr_name, None)):
                    continue
                    
                try:
                    attr = getattr(obj, attr_name)
                    current_path = f"{path}.{attr_name}"
                    
                    # 检查当前属性是否是beta_head
                    if attr_name == "beta_head":
                        print(f"[DEBUG] 找到beta_head: {current_path}")
                        return current_path, attr
                        
                    # 递归检查子对象，但跳过基本类型和常见的大型对象
                    if (isinstance(attr, object) and 
                        not isinstance(attr, (str, int, float, bool, list, dict, set, tuple)) and
                        attr_name not in ('state', 'optimizer', 'scheduler')):
                        result_path, result_obj = find_beta_head(attr, current_path, max_depth - 1)
                        if result_path:
                            return result_path, result_obj
                except:
                    # 忽略访问错误
                    pass
                    
            return None, None
            
        beta_head_location, beta_head = find_beta_head(trainer)
        
    print(f"[DEBUG] beta_head位置: {beta_head_location}")
    
    # 如果找到beta_head，将其设置到监控回调中
    if beta_head is not None:
        print("[DEBUG] 将找到的beta_head设置到监控回调中")
        beta_monitor.set_beta_head(beta_head)
    
    if beta_head is not None:
        print("训练前beta_head参数:")
        for name, param in beta_head.named_parameters():
            print(f"  {name}: {param.mean()}, requires_grad={param.requires_grad}")
        # 打印beta_head的参数
    
    # 阶段4: 执行训练
    print('=' * 60)
    print("开始训练...")
    print('=' * 60)
    
    # 执行训练
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # 训练后
    if hasattr(trainer, "beta_head"):
        print("训练后beta_head参数:")
        for name, param in trainer.beta_head.named_parameters():
            print(f"  {name}: {param.mean()}, requires_grad={param.requires_grad}")
        # 打印beta_head的参数
        
    # 保存模型
    print("保存模型...")
    trainer.save_model()
    trainer.save_state()
    
    # 记录指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行训练并监控beta_head参数')
    parser.add_argument('--config', type=str, help='训练配置文件路径')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb项目名称')
    args = parser.parse_args()

    # --config 如果为空, 选择默认配置
    default_config = 'ledpo_progressive_dev/qwen15_lora_foodpo.yaml'
    if args.config is None:
        args.config = default_config
    
    # 运行训练并监控
    print(f"正在运行训练并监控...")
    # --wandb_project 如果为空, 不使用wandb
    if args.wandb_project is None:
        run_with_monitor(args.config)
    else:
        run_with_monitor(args.config, args.wandb_project)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 