---
title: LLaMA Factory: Unified Efficient Fine-Tuning of 100+ Language Models
created: 2026-04-23
updated: 2026-04-23
type: paper
tags: [arxiv, framework, lora, qlora, peft, fine-tuning, acl-2024]
sources: [raw/papers/arxiv-2403.13372.md, raw/articles/voltagent-llama-factory.md]
arxiv: https://arxiv.org/abs/2403.13372
github: https://github.com/hiyouga/LLaMA-Factory
stars: 25000+
---

# LLaMA Factory

## 概述

LlamaFactory 是一个统一的高效微调框架，支持 100+ 大语言模型的灵活定制微调，无需编码即可通过 Web UI 完成。

**核心价值**：
- 统一框架整合多种高效训练方法
- 支持 100+ LLM 和 VLM 模型
- 零代码 Web UI (LlamaBoard)
- 内存优化：从 18 bytes/param 降至 0.6 bytes/param

## 论文信息

- **标题**: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models
- **作者**: Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, Yongqiang Ma
- **发表**: ACL 2024
- **arXiv**: [2403.13372](https://arxiv.org/abs/2403.13372)
- **GitHub**: 25,000+ stars, 3,000+ forks

## 框架架构

三大核心模块：

```
┌─────────────────────────────────────────────────────┐
│                  LlamaBoard (Web UI)                │
│              零代码配置与监控界面                     │
└─────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────┐
│                  LlamaFactory Core                  │
├─────────────────┬─────────────────┬─────────────────┤
│  Model Loader   │  Data Worker    │    Trainer      │
│  模型加载器      │  数据处理器      │   训练器        │
│  - 初始化        │  - 单轮对话      │  - 预训练       │
│  - 模型补丁      │  - 多轮对话      │  - 指令微调     │
│  - 量化         │  - 数据流水线    │  - 偏好优化     │
│  - 适配器挂载    │                 │                │
└─────────────────┴─────────────────┴─────────────────┘
```

## 支持的高效训练技术

### 高效优化方法

| 方法 | 类型 | 说明 |
|------|------|------|
| [[lora\|LoRA]] | PEFT | 低秩适配，冻结权重注入可训练矩阵 |
| [[qlora\|QLoRA]] | PEFT+量化 | 4-bit 量化 + LoRA |
| Full-tuning | 全量 | 全参数微调 |
| Freeze-tuning | 部分冻结 | 冻结部分层 |
| GaLore | 优化器 | Gradient Low-Rank Projection |

### 高效计算方法

| 方法 | 作用 |
|------|------|
| 4-bit/8-bit 量化 | 降低显存占用 |
| FlashAttention-2 | 加速注意力计算 |
| Unsloth | 训练加速 |
| DeepSpeed ZeRO | 分布式训练优化 |
| FSDP | 全分片数据并行 |

### 内存优化效果

| 训练方式 | 内存/参数 |
|----------|-----------|
| 混合精度 (fp16) | 18 bytes |
| 半精度 (fp16) | 8 bytes |
| LlamaFactory 优化后 | **0.6 bytes** |

## 支持的模型

### 大语言模型 (LLM)

| 模型系列 | 代表模型 |
|----------|----------|
| Llama | Llama 2, Llama 3, Llama 4 |
| Qwen | Qwen, Qwen2, Qwen2.5, Qwen3 |
| Mistral | Mistral-7B, Mixtral |
| Yi | Yi-6B, Yi-34B |
| ChatGLM | ChatGLM3-6B |
| Gemma | Gemma-2B, Gemma-7B |
| DeepSeek | DeepSeek-R1 |
| InternLM | Intern-S1-mini |
| GLM | GLM-4.1V |

### 多模态模型 (VLM)

| 模型 | 类型 |
|------|------|
| InternVL3 | 视觉语言 |
| MiniCPM-V | 视觉语言 |
| Qwen2-Audio | 音频 |
| Qwen2.5-Omni | 全模态 |

### 2025 年新增支持

- **OFT/OFTv2**: 正交微调 (Aug 2025)
- **GPT-OSS**: 开源 GPT 模型
- **Muon/APOLLO**: 新优化器
- **SGLang**: 推理后端

## 训练方式

### 1. 命令行 (CLI)

```bash
# 克隆仓库
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 安装
pip install -e ".[torch,metrics]"

# LoRA 微调 Llama 3
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

# DPO 对齐 Mistral
llamafactory-cli train examples/train_dpo/mistral_dpo.yaml
```

### 2. Web UI (LlamaBoard)

```bash
llamafactory-cli webui
```

功能：
- 模型选择与配置
- 数据集管理
- 训练参数调优
- 实时训练监控
- 模型测试对话

### 3. 训练后操作

```bash
# 测试模型
llamafactory-cli chat your_model.yaml

# 导出/合并 LoRA
llamafactory-cli export your_config.yaml
```

## 数据格式

支持的数据源：
- 本地 JSON 文件
- Hugging Face 数据集
- ModelScope Hub

数据结构：
- 单轮对话
- 多轮对话
- 偏好数据 (DPO/RLHF)

## 实验结果

### 训练效率 (PubMed 数据集)

| 模型 | 方法 | 显存 | PPL |
|------|------|------|-----|
| Gemma-2B | Full | 高 | 最佳 |
| Gemma-2B | LoRA | 低 | 接近 |
| Gemma-2B | QLoRA | **最低** | 接近 |

### 下游任务性能 (ROUGE)

| 模型         | LoRA   | QLoRA | GaLore |
| ---------- | ------ | ----- | ------ |
| Llama3-8B  | **最佳** | 接近    | 中      |
| Mistral-7B | 最佳     | 最佳    | 中      |
| Yi-6B      | 最佳     | 接近    | 中      |

**结论**：LoRA 和 QLoRA 在大多数任务上效果最佳。

## 与其他框架对比

| 特性 | LlamaFactory | Axolotl | TRL | Unsloth |
|------|--------------|---------|-----|---------|
| 模型支持 | **100+** | ~50 | ~30 | ~20 |
| Web UI | ✅ LlamaBoard | ❌ | ❌ | ❌ |
| LoRA | ✅ | ✅ | ✅ | ✅ |
| QLoRA | ✅ | ✅ | ✅ | ✅ |
| DPO | ✅ | ✅ | ✅ | ❌ |
| RLHF (PPO) | ✅ | ✅ | ✅ | ❌ |
| 多模态 | ✅ | 部分 | ❌ | ❌ |
| 量化训练 | ✅ | ✅ | 部分 | ✅ |

## 适用场景

### 推荐 LlamaFactory 的情况
- 需要快速实验多种模型
- 团队需要 Web UI 降低门槛
- 支持多模态模型微调
- 需要最新的训练方法集成

### 可选其他框架
- **Axolotl**: YAML 配置深度定制
- **TRL**: 深度 RLHF 研究
- **Unsloth**: 极致速度优化

## 相关资源

- **文档**: https://llamafactory.readthedocs.io
- **GitHub**: https://github.com/hiyouga/LLaMA-Factory
- **论文**: ACL 2024

## 相关内容

- [[lora|LoRA 详解]]
- [[qlora|QLoRA 详解]]
- [[dpo|DPO 详解]]
- [[rlhf|RLHF 详解]]
- [[peft-methods-comparison|PEFT 方法对比]]
