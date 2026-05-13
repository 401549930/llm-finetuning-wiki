---
title: Unsloth: 极速大模型微调
created: 2026-04-23
updated: 2026-04-23
type: entity
tags: [unsloth, framework, lora, qlora, speed-optimization]
github: https://github.com/unslothai/unsloth
stars: 35000+
---

# Unsloth

## 概述

Unsloth 是一个专注于训练速度优化的微调框架，通过手动推导反向传播梯度、自定义 CUDA 内核和 4-bit 量化优化，实现 2-5 倍的训练加速和 80% 的显存节省。

**核心价值**：
- 2-5x 训练加速（无精度损失）
- 80% 显存节省
- 零代码变更，兼容 HuggingFace 生态
- 支持主流开源模型

## 加速原理

### 1. 手动反向传播
```python
# 传统方式：自动微分 (PyTorch autograd)
# 计算图开销大，内存占用高

# Unsloth 方式：手动推导梯度
# 直接编写 LoRA 的反向传播代码
# 消除计算图中间状态，节省内存和计算
```

### 2. 自定义 CUDA 内核
- 融合矩阵乘法 + LoRA 适配器计算
- 4-bit 量化解压与计算融合
- 减少内核启动开销和显存搬运

### 3. 量化优化
- 支持 4-bit QLoRA 训练
- 优化 bitsandbytes 量化流程
- 支持 16-bit LoRA 训练

## 速度对比

| 模型 | 场景 | HuggingFace | Unsloth | 加速比 |
|------|------|-------------|---------|--------|
| Llama-3-8B | SFT LoRA | ~1.5 it/s | ~3.5 it/s | **2.3x** |
| Mistral-7B | SFT QLoRA | ~1.2 it/s | ~3.0 it/s | **2.5x** |
| Llama-3-70B | QLoRA | ~0.2 it/s | ~0.5 it/s | **2.5x** |

### 显存对比 (Llama-3-8B, seq_len=2048)

| 配置 | HuggingFace | Unsloth | 节省 |
|------|-------------|---------|------|
| LoRA (fp16) | ~20GB | ~12GB | **40%** |
| QLoRA (4bit) | ~8GB | ~5GB | **37%** |

## 支持的模型

| 模型系列 | 支持状态 |
|----------|----------|
| Llama 2/3/3.1/3.2/4 | ✅ 完全优化 |
| Mistral / Mixtral | ✅ 完全优化 |
| Qwen2 / Qwen2.5 | ✅ 完全优化 |
| Gemma / Gemma2 | ✅ 完全优化 |
| Phi-3 / Phi-4 | ✅ 完全优化 |
| DeepSeek-R1 | ✅ 支持 |
| Yi | ✅ 支持 |
| CodeLlama | ✅ 支持 |

## 快速开始

### 安装
```bash
pip install unsloth
# 或从源码
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### SFT 微调
```python
from unsloth import FastLanguageModel
from trl import SFTTrainer

# 加载模型（自动优化）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 添加 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

# 训练（兼容 HuggingFace Trainer）
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
)
trainer.train()
```

### DPO 微调
```python
from unsloth import FastLanguageModel
from trl import DPOTrainer

model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(...)

trainer = DPOTrainer(
    model=model,
    train_dataset=preference_dataset,
    peft_config=None,  # LoRA 已挂载
)
trainer.train()
```

## 与其他框架对比

| 特性 | Unsloth | [[axolotl\|Axolotl]] | [[llama-factory\|LLaMA Factory]] | [[trl\|TRL]] |
|------|---------|---------|---------|-----|
| 训练速度 | **最快** | 标准 | 标准 | 标准 |
| 显存优化 | **最佳** | 好 | 好 | 标准 |
| 对齐方法 | SFT/DPO | **最全** | 广 | **最全** |
| DeepSpeed | ❌ | ✅ | ✅ | 部分 |
| 多GPU | ❌ (单卡) | ✅ | ✅ | ✅ |
| 自定义 | 低 | 中 | 低 | **最高** |
| 上手难度 | **最低** | 中 | 低 | 高 |

## 适用场景

### 推荐 Unsloth 的情况
- 单卡消费级 GPU 微调（如 RTX 3090/4090）
- 快速原型实验和迭代
- SFT 或 DPO 微调任务
- 显存受限但需要训练大模型

### 不推荐的情况
- 多卡/多节点分布式训练 → 用 [[axolotl|Axolotl]]
- 需要 PPO/GRPO 等复杂 RL 方法 → 用 [[trl|TRL]]
- 需要零代码 Web UI → 用 [[llama-factory|LLaMA Factory]]

## 相关资源

- GitHub: https://github.com/unslothai/unsloth
- 文档: https://docs.unsloth.ai
- 模型: https://huggingface.co/unsloth

## 相关内容

- [[lora|LoRA]] — Unsloth 核心加速的PEFT方法
- [[qlora|QLoRA]] — 4-bit量化+LoRA
- [[llama-factory|LLaMA Factory]] — 对比框架
- [[axolotl|Axolotl]] — 对比框架
- [[trl|TRL]] — 对比框架
