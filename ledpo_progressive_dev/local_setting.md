# 本地设置

conda 环境名: test

项目根目录: /root/demo/llama_factory

本地模型目录: /root/models/
```bash
(test) (base) root@intern-studio-50002342:~/demo/llama_factory# ls /root/models/
DeepSeek-R1-Distill-Qwen-1.5B  Shanghai_AI_Laboratory  bce-reranker-base_v1       internlm2_5-7b-chat                              temp
Qwen1.5-0.5B                   backup                  internlm-xcomposer2-vl-7b  qwen                                             tmp.py
Qwen1.5-0.5B-Chat              baicai003               internlm2-chat-7b          speech_sambert-hifigan_tts_zhiyan_emo_zh-cn_16k
Qwen2.5-1.5B-Instruct          bce-embedding-base_v1   internlm2_5-1_8b-chat      tclf90
```
请优先选择 Qwen1.5-0.5B 进行测试. 

本地数据目录: data/ 
```bash
(test) (base) root@intern-studio-50002342:~/demo/llama_factory# ls data/
README.md            alpaca_zh_demo.json  dataset_info.json  glaive_toolcall_en_demo.json  identity.json         mllm_demo.json        ultra_chat
README_zh.md         belle_multiturn      dpo_en_demo.json   glaive_toolcall_zh_demo.json  kto_en_demo.json      mllm_demo_data        wiki_demo.txt
alpaca_en_demo.json  c4_demo.json         dpo_zh_demo.json   hh_rlhf_en 
```
请有限选择 dpo_en_demo.json 进行测试. 

实验结果目录: results/
