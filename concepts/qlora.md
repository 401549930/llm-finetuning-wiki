---
title: QLoRA: Quantized Low-Rank Adaptation
created: 2026-04-23
updated: 2026-04-23
type: method
tags: [qlora, lora, quantization, peft]
sources: [raw/papers/arxiv-2305.14314.md]
arxiv_id: "2305.14314"
---

# QLoRA: Quantized Low-Rank Adaptation

## 一句话总结
在4-bit量化模型上应用LoRA，实现单卡微调65B模型。

## 核心原理

QLoRA 将 LoRA 与量化结合：
1. 将预训练模型量化为 4-bit
2. 在量化模型上训练 LoRA adapter
3. 梯度反向传播通过量化模型到 LoRA 参数

### 三大创新

#### 1. 4-bit NormalFloat (NF4)
- 信息论最优的 4-bit 数据类型
- 适用于正态分布的权重
- 比标准 4-bit 浮点更精确

#### 2. 双重量化 (Double Quantization)
- 对量化常数再次量化
- 进一步减少显存占用
- 每参数平均节省 0.37 bit

#### 3. 分页优化器 (Paged Optimizers)
- 使用 NVIDIA Unified Memory
- GPU 显存不足时自动转移到 CPU
- 避免显存溢出错误

## 实现要点

```python
# 使用 bitsandbytes 实现
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b",
    quantization_config=bnb_config,
    device_map="auto"
)

# 添加 LoRA adapter
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
```

## 显存对比

| 模型大小 | 全量微调 | LoRA | QLoRA |
|----------|----------|------|-------|
| 7B | ~28GB | ~16GB | ~6GB |
| 13B | ~52GB | ~28GB | ~10GB |
| 33B | ~132GB | ~68GB | ~24GB |
| 65B | ~260GB | ~130GB | ~48GB |

## Guanaco 模型

QLoRA 论文训练的模型系列：
- Guanaco-65B：在 Vicuna benchmark 上达到 ChatGPT 99.3% 性能
- 仅需 24 小时单卡训练

## 优缺点

| 优点 | 缺点 |
|------|------|
| 单卡微调 65B 模型 | 4-bit 精度有损失 |
| 显存降低 75%+ | 训练速度略慢于 LoRA |
| 效果接近全量微调 | 需要兼容的 GPU |
| 完全保留原模型 | - |

## 适用场景

- 消费级 GPU 微调大模型
- 原型开发和实验
- 显存受限场景
- 需要快速验证想法

## 相关方法

- [[lora|LoRA]]：基础方法
- [[gguf|GGUF]]：另一种量化格式
- [[gptq|GPTQ]]：另一种量化方法

## 主要论文

- [[raw/papers/arxiv-2305.14314|QLoRA: Efficient Finetuning of Quantized LLMs]] (Dettmers et al., 2023)

## 参考文献

- GitHub: https://github.com/artidoro/qlora
- Hugging Face: https://huggingface.co/docs/peft/conceptual_guides/qlora
