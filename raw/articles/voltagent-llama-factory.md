---
title: VoltAgent - LLaMA Factory 介绍
created: 2026-04-23
updated: 2026-04-23
type: source
tags: [article, tutorial, framework]
source: https://voltagent.dev/blog/llama-factory/
---

# What is LLaMA Factory? LLM Fine-Tuning

**来源**: VoltAgent Blog
**日期**: 2025-10-14

## 概述

VoltAgent 对 LLaMA Factory 的详细介绍，包括 2025 年最新更新。

## 2025 年新增功能

### 新模型支持
- OFT/OFTv2 (Aug 2025) - 正交微调
- Intern-S1-mini (Aug 2025)
- GPT-OSS (Aug 2025)
- GLM-4.1V, Qwen3, InternVL3, Llama 4
- Qwen2.5-Omni, Qwen2-Audio, DeepSeek-R1

### 新优化器
- Muon
- APOLLO

### 推理后端
- SGLang

## 快速开始

```bash
# 克隆
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 安装
pip install -e ".[torch,metrics]"

# Web UI
llamafactory-cli webui

# 训练
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

## 数据准备

- 支持 JSON 格式
- Hugging Face 数据集
- ModelScope Hub
- 配置文件: `data/dataset_info.json`

## 生产部署

LlamaFactory 包含部署 API，但高流量生产环境可能需要额外的 MLOps 工具。
