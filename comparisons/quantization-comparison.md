---
title: 量化方法对比
created: 2026-04-23
updated: 2026-04-23
type: comparison
tags: [comparison, gguf, gptq, awq, quantization, bnb]
sources: []
---

# 量化方法对比

## 对比概览

| 方法 | 用途 | 需要校准 | CPU推理 | GPU推理 | 支持训练 |
|------|------|----------|---------|---------|----------|
| [[gguf\|GGUF]] | 推理 | ❌ | ✅ | 部分 | ❌ |
| [[gptq\|GPTQ]] | 推理 | ✅ | ❌ | ✅ | ❌ |
| [[awq\|AWQ]] | 推理 | ✅ | ❌ | ✅ | ❌ |
| bitsandbytes | 训练+推理 | ❌ | ❌ | ✅ | ✅ |

## 详细对比

### 量化方法原理

| 方法 | 核心原理 | 创新点 |
|------|----------|--------|
| GGUF | 直接量化 + mmap | CPU 优化 |
| GPTQ | Hessian补偿逐行量化 | 二阶信息补偿 |
| AWQ | 激活感知通道缩放 | 保护重要权重 |
| BnB | 动态量化 + 离群值分离 | 无校准即时量化 |

### 精度对比 (Llama-2-7B, Wiki2 Perplexity)

| 方法 | 4-bit | 8-bit |
|------|-------|-------|
| FP16 基线 | 5.47 | 5.47 |
| GGUF Q4_K_M | 5.65 | 5.52 |
| GPTQ-4bit | **5.63** | 5.50 |
| AWQ-4bit | **5.58** | 5.49 |
| BnB NF4 | 5.60 | 5.51 |

AWQ 4-bit 精度最优。

### 量化速度

| 模型大小 | GGUF | GPTQ | AWQ | BnB |
|----------|------|------|-----|-----|
| 7B | ~2min | ~15min | ~5min | **即时** |
| 13B | ~4min | ~30min | ~10min | **即时** |
| 70B | ~15min | ~2h | ~40min | **即时** |

### 推理速度 (7B, A100)

| 方法 | 吞吐量 (token/s) | 首 token 延迟 |
|------|------------------|---------------|
| FP16 | ~30 | ~50ms |
| GGUF (Q4) CPU | ~15 | ~200ms |
| GGUF (Q4) GPU | ~40 | ~60ms |
| GPTQ-4bit | ~45 | ~55ms |
| AWQ-4bit | **~55** | **~50ms** |
| BnB-4bit | ~35 | ~60ms |

### 显存需求 (7B 模型)

| 方法 | 4-bit 模型大小 | 推理显存 |
|------|---------------|----------|
| FP16 | 14GB | 16GB |
| GGUF Q4_K_M | ~4GB | ~5GB (CPU内存) |
| GPTQ-4bit | ~3.9GB | ~6GB |
| AWQ-4bit | ~3.9GB | ~5.5GB |
| BnB NF4 | ~3.5GB | ~5GB |

## 按场景选择

### 本地桌面推理

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| Mac M系列 | **GGUF** | Metal 加速支持好 |
| 纯 CPU | **GGUF** | 唯一选择 |
| 消费级 GPU (8-12GB) | GGUF/AWQ | 部分GPU卸载 |
| 消费级 GPU (24GB) | AWQ/GPTQ | 全GPU推理 |

### 服务器部署

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| 高吞吐 vLLM | **AWQ/GPTQ** | GPU 利用率最高 |
| 多模型服务 | **AWQ** | 精度+速度平衡 |
| CPU 服务器 | **GGUF** | 唯一选择 |
| 混合部署 | **GGUF** | GPU卸载灵活 |

### 训练场景

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| QLoRA 微调 | **bitsandbytes** | 唯一支持训练的 |
| SFT 训练 | BnB (4bit) | NF4 效果最好 |
| DPO/KTO | BnB + LoRA | 标准组合 |
| RLHF/PPO | BnB + LoRA | 标准组合 |

## 量化精度选择指南

### GGUF 精度选择

| 显存/内存 | 推荐精度 |
|-----------|----------|
| 充裕 (>2x模型大小) | Q8_0 |
| 适中 | **Q5_K_M** |
| 紧张 | Q4_K_M |
| 极限 | Q3_K_M |
| 不推荐 | Q2_K |

### GPTQ/AWQ 精度选择

| 需求 | 推荐精度 |
|------|----------|
| 追求质量 | 8-bit |
| 平衡 | **4-bit-128g** |
| 极限压缩 | 3-bit-128g |

## 格式互转

### 常见转换路径
```
HuggingFace (FP16)
  ├──→ GGUF (llama.cpp convert)
  ├──→ GPTQ (AutoGPTQ)
  ├──→ AWQ (AutoAWQ)
  └──→ BnB (运行时量化，无需转换)

GPTQ ←→ AWQ (需反量化再量化，不推荐)
```

### 推荐直接用预量化模型
- TheBloke: https://huggingface.co/TheBloke (GPTQ/AWQ/GGUF)
- LM Studio: 内置 GGUF 模型库
- Ollama: 内置 GGUF 模型库

## 组合使用

### 训练 → 推理 全流程
```
1. 用 BnB NF4 + LoRA 训练 (QLoRA)
2. 合并 LoRA 权重
3. 转换为 GGUF/GPTQ/AWQ 部署
```

### 多格式部署
```
同一模型:
  ├── GGUF  → 本地/边缘部署
  ├── AWQ   → GPU 服务器部署
  └── GPTQ  → 高精度 GPU 部署
```

## 相关内容

- [[gguf|GGUF]] — CPU推理量化格式
- [[gptq|GPTQ]] — GPU高精度量化
- [[awq|AWQ]] — 激活感知量化
- [[qlora|QLoRA]] — 训练时量化
- [[bitsandbytes]] — 动态量化库
- [[lora|LoRA]] — 常与量化组合
