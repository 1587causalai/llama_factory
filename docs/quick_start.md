# 快速开始

本文档专注于帮助您最快速地开始使用 LLaMA Factory 进行模型微调，省略了许多高级特性的细节。

## 安装

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch]"
```

## 可视化界面微调

最简单的方式是使用可视化界面：

```bash
llamafactory-cli webui
```

1. 打开浏览器访问 `http://localhost:7860`
2. 在界面中选择"训练"标签页
3. 选择要微调的模型和训练方法
4. 上传或选择训练数据
5. 点击"开始训练"

## 本地模型快速微调

如果您已经下载了模型权重到本地，可以直接使用本地路径进行微调。

### 1. 准备模型

假设您的模型存放在以下路径：
```
/path/to/model/
├── config.json
├── tokenizer.model
├── tokenizer_config.json
└── pytorch_model.bin (或 adapter_*.bin)
```

### 2. 准备数据

创建一个简单的 JSON 格式训练数据（例如 `data.json`）：

```json
[
    {
        "instruction": "请介绍一下北京",
        "output": "北京是中国的首都，有着悠久的历史文化..."
    }
]
```

### 3. 开始微调

使用以下命令进行 LoRA 微调：

```bash
llamafactory-cli train \
    --model_name_or_path /path/to/model \
    --dataset data.json \
    --template default \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --output_dir output
```

### 4. 测试效果

```bash
llamafactory-cli chat \
    --model_name_or_path /path/to/model \
    --adapter_name_or_path output \
    --template default
```

## 常见问题

1. **显存不足？**
   - 使用 QLoRA 训练：添加参数 `--quantization_bit 4`
   - 减小 batch size：添加参数 `--per_device_train_batch_size 1`

2. **训练很慢？**
   - 启用 Flash Attention 2：添加参数 `--flash_attn fa2`
   - 使用 unsloth 优化：添加参数 `--use_unsloth true`

3. **模型不会说中文？**
   - 选择合适的中文模型，如 Baichuan、ChatGLM、Qwen 等
   - 使用中文指令数据集进行微调

## 下一步

- 查看[完整文档](README_zh.md)了解更多高级特性
- 参考[示例配置](examples/README_zh.md)了解更多训练选项
- 加入[微信群](assets/wechat.jpg)获取帮助 