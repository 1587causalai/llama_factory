# 开发日志






## 2025-01-03 HuggingFace 缓存路径说明

在使用 LLaMA Factory 进行 DPO 训练时,发现一些重要的缓存路径信息:

### 1. 模型文件缓存
默认路径: `/root/.cache/huggingface/hub/`
- 包含模型配置文件(config.json)
- 分词器配置(tokenizer_config.json)
- 模型权重文件等

可通过设置环境变量修改:
```bash
export TRANSFORMERS_CACHE="/path/to/your/directory"
```

### 2. 数据集缓存
默认路径: `/root/.cache/huggingface/datasets/`
- 包含数据集元数据
- Arrow或Parquet格式的实际数据文件

修改方式:
1. 通过环境变量:
```bash
export HF_DATASETS_CACHE="/path/to/your/directory"
```

2. 通过代码指定:
```python
dataset = load_dataset("dataset_name", cache_dir="/path/to/your/directory")
```

### 3. Triton缓存
默认路径: `/root/.triton/autotune`

可通过设置环境变量修改:
```bash
export TRITON_CACHE_DIR="/path/to/your/directory"
```
