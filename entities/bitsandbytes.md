---
title: bitsandbytes: 模型量化库
created: 2026-04-23
updated: 2026-04-23
type: entity
tags: [bnb, quantization, gguf, gptq, awq, 4bit, 8bit]
github: https://github.com/TimDettmers/bitsandbytes
stars: 7000+
developer: Tim Dettmers (华盛顿大学)
---

# bitsandbytes

## 概述

bitsandbytes 是大语言模型量化的基础库，提供 8-bit 和 4-bit 量化功能。它是 [[qlora|QLoRA]] 的核心依赖，也是 HuggingFace Transformers 量化推理的标准后端。

**核心价值**：
- 零代码量化，2行代码即可加载量化模型
- QLoRA 训练的标准后端
- 显存节省 4-8x
- 与 HuggingFace 生态深度集成

## 量化方法

### 8-bit 量化 (LLM.int8())
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    load_in_8bit=True,  # 一行启用 8-bit 量化
)
# 显存: 14GB → ~7GB
```

**原理**：
- 大部分权重用 8-bit 存储
- 离群值特征维度保留 FP16 (约 0.1%)
- 混合精度分解保证精度

### 4-bit 量化 (QLoRA 核心)
```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # 量化类型
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度
    bnb_4bit_use_double_quant=True,     # 双重量化
)
# 显存: 14GB → ~3.5GB
```

## 4-bit 量化类型

| 类型 | 全称 | 说明 | 适用场景 |
|------|------|------|----------|
| `nf4` | NormalFloat 4 | 正态分布优化的4-bit | **微调训练 (推荐)** |
| `fp4` | Float 4 | 标准4-bit浮点 | 推理 |
| `int4` | Integer 4 | 整数量化 | 边缘部署 |

## 双重量化 (Double Quantization)

```
普通4-bit:
  权重 → 4-bit 量化 → 量化常数 (FP32)

双重量化:
  权重 → 4-bit 量化 → 量化常数 → 再量化为 FP4
  节省: ~0.37 bits/param (65B模型省约3GB)
```

## 显存对比

以 Llama-65B 为例：

| 方式 | 显存 | 说明 |
|------|------|------|
| FP32 | ~260GB | 全精度 |
| FP16 | ~130GB | 半精度 |
| 8-bit | ~65GB | bitsandbytes |
| 4-bit (NF4) | ~33GB | bitsandbytes |
| 4-bit + 双重 | ~30GB | bitsandbytes |

## 与其他量化方案对比

| 特性 | bitsandbytes | [[gguf\|GGUF]] | [[gptq\|GPTQ]] | [[awq\|AWQ]] |
|------|-------------|------|------|------|
| 量化方式 | 动态 | 静态 | 静态 | 静态 |
| 需要校准数据 | ❌ | ❌ | ✅ | ✅ |
| 支持训练 | ✅ (QLoRA) | ❌ | ❌ | ❌ |
| 支持推理 | ✅ | ✅ | ✅ | ✅ |
| GPU 推理 | ✅ | 部分 | ✅ | ✅ |
| CPU 推理 | ❌ | ✅ | ❌ | ❌ |
| 量化速度 | **最快** (即时) | 快 | 慢 | 中 |
| 精度 | 好 | 好 | **最佳** | 好 |

## 在微调框架中的使用

### Axolotl
```yaml
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bf16
bnb_4bit_use_double_quant: true
```

### LLaMA Factory
```yaml
quantization_bit: 4
quantization_method: bitsandbytes
```

### TRL
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config
)
```

## 安装

```bash
# CUDA 11.x
pip install bitsandbytes

# CUDA 12.x
pip install bitsandbytes>=0.43.0

# 验证
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

## 相关资源

- GitHub: https://github.com/TimDettmers/bitsandbytes
- 论文: LLM.int8() (NeurIPS 2022)
- 论文: QLoRA (NeurIPS 2023)

## 相关内容

- [[qlora|QLoRA]] — 基于 bitsandbytes 的4-bit微调
- [[gguf|GGUF]] — CPU推理量化格式
- [[gptq|GPTQ]] — 另一种GPU量化方案
- [[awq|AWQ]] — 激活感知量化
- [[lora|LoRA]] — 常与量化组合使用
