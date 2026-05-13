---
title: GGUF: GPT-Generated Unified Format
created: 2026-04-23
updated: 2026-04-23
type: concept
tags: [gguf, quantization, inference, cpu-inference, llama-cpp]
sources: []
---

# GGUF: GPT-Generated Unified Format

## 一句话总结
为 llama.cpp 设计的模型量化格式，支持 CPU/GPU 混合推理，是大模型本地部署的事实标准。

## 核心原理

GGUF 是 GGML 格式的后继者，解决了 GGML 的向后兼容性问题，并增加了更多量化选项和元数据支持。

### 与 GGML 的区别

| 方面 | GGML | GGUF |
|------|------|------|
| 向后兼容 | 每次改格式都断 | **KV 元数据，可扩展** |
| 量化选项 | 少 | **丰富** (2-8 bit) |
| 元数据 | 有限 | **完整** (超参/词表) |
| 加载速度 | 中 | **更快** (mmap) |

## 量化类型

### 量化精度层级

| 类型 | 位数 | 大小 (7B) | 质量 | 速度 | 推荐场景 |
|------|------|-----------|------|------|----------|
| Q8_0 | 8-bit | ~7GB | ★★★★★ | ★★★ | 需要高精度 |
| Q6_K | 6-bit | ~5.5GB | ★★★★☆ | ★★★★ | 性价比高 |
| Q5_K_M | 5-bit | ~4.7GB | ★★★★ | ★★★★ | **推荐默认** |
| Q5_0 | 5-bit | ~4.5GB | ★★★☆ | ★★★★ | 内存紧张 |
| Q4_K_M | 4-bit | ~4.0GB | ★★★★ | ★★★★★ | **最常用** |
| Q4_0 | 4-bit | ~3.8GB | ★★★ | ★★★★★ | 内存极限 |
| Q3_K_M | 3-bit | ~3.2GB | ★★☆ | ★★★★★ | 极端节省 |
| Q2_K | 2-bit | ~2.5GB | ★★ | ★★★★★ | 不推荐 |

### K-量化说明

K-量化 (如 Q4_K_M) 使用混合精度：
- 重要层用更高精度
- 不重要层用更低精度
- 在同等体积下质量更好

后缀含义：
- `_S`: Small (最小体积)
- `_M`: Medium (推荐)
- `_L`: Large (更好质量)

## 转换流程

### HuggingFace → GGUF
```bash
# 安装 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# 转换模型
python convert_hf_to_gguf.py \
    /path/to/hf-model \
    --outfile model.gguf \
    --outtype f16

# 量化
./llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M
./llama-quantize model.gguf model-Q5_K_M.gguf Q5_K_M
./llama-quantize model.gguf model-Q8_0.gguf Q8_0
```

### 使用 ollama
```bash
# 直接从 HuggingFace 拉取
ollama run hf.co/username/model:Q4_K_M

# 或创建 Modelfile
echo "FROM ./model-Q4_K_M.gguf" > Modelfile
ollama create mymodel -f Modelfile
ollama run mymodel
```

## 推理引擎

| 引擎 | 特点 | GGUF 支持 |
|------|------|-----------|
| llama.cpp | 原生，最全功能 | ✅ 创建者 |
| Ollama | 易用，一键部署 | ✅ 基于 llama.cpp |
| LM Studio | GUI，适合终端用户 | ✅ |
| text-generation-webui | Web UI | ✅ |
| GPT4All | 跨平台桌面 | ✅ |

## GPU 卸载

GGUF 支持部分层在 GPU 运行，其余在 CPU：

```bash
# 全 CPU
./llama-cli -m model-Q4_K_M.gguf

# GPU 卸载 20 层
./llama-cli -m model-Q4_K_M.gguf -ngl 20

# 全 GPU
./llama-cli -m model-Q4_K_M.gguf -ngl 999
```

## 内存需求估算

| 模型 | Q4_K_M | Q5_K_M | Q8_0 |
|------|--------|--------|------|
| 7B | ~4GB | ~5GB | ~7GB |
| 13B | ~7GB | ~9GB | ~13GB |
| 34B | ~19GB | ~23GB | ~34GB |
| 70B | ~39GB | ~47GB | ~70GB |

## 与其他量化方案对比

| 特性 | GGUF | [[gptq\|GPTQ]] | [[awq\|AWQ]] | bitsandbytes |
|------|------|------|------|-------------|
| 用途 | 推理 | 推理 | 推理 | 训练+推理 |
| CPU 推理 | ✅ | ❌ | ❌ | ❌ |
| GPU 推理 | ✅ | ✅ | ✅ | ✅ |
| 需要校准 | ❌ | ✅ | ✅ | ❌ |
| 量化速度 | 快 | 慢 | 中 | 即时 |
| 部署便利 | **最佳** | 好 | 好 | 一般 |

## 适用场景

- 本地 CPU 推理
- 消费级 GPU 部署
- 内存受限环境
- 快速部署 (Ollama)
- 离线环境

## 相关资源

- llama.cpp: https://github.com/ggerganov/llama.cpp
- GGUF 规范: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Ollama: https://ollama.ai

## 相关内容

- [[gptq|GPTQ]] — GPU 量化方案
- [[awq|AWQ]] — 激活感知量化
- [[qlora|QLoRA]] — 训练时量化
- [[lora|LoRA]] — 常与量化组合
