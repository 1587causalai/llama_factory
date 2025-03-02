from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载基础模型和tokenizer
model_path = '/Users/gongqian/models/Qwen1.5-0.5B'
adapter_path = 'output/qwen_foodpo_test'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='mps',
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 加载LoRA适配器
from peft import PeftModel
model = PeftModel.from_pretrained(model, adapter_path)

# 设置生成参数
prompt = '''我想让你帮我写一个简短的Python函数，用于计算斐波那契数列的第n项。
要求：
1. 使用递归方法
2. 添加适当的注释
3. 包含一个简单的测试用例
请提供完整的代码。'''

inputs = tokenizer(prompt, return_tensors='pt').to('mps')

# 生成回复
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,  # 启用采样模式
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print('输入:', prompt)
print('输出:', response) 