---
title: Axolotl: 高效大模型微调框架
created: 2026-04-23
updated: 2026-04-23
type: entity
tags: [axolotl, framework, lora, qlora, dpo, rlhf, deepspeed]
github: https://github.com/OpenAccess-AI-Collective/axolotl
stars: 8000+
---

# Axolotl

## 概述

Axolotl 是一个专注于大语言模型微调的开源训练框架，以 YAML 配置驱动，支持广泛的训练方法和模型架构。设计理念是"一个配置文件完成所有训练"。

**核心价值**：
- 纯 YAML 配置，无需写代码
- 支持最全的训练方法组合
- 深度集成 DeepSpeed 分布式训练
- 活跃社区持续更新

## 框架架构

```
┌─────────────────────────────────────┐
│           YAML 配置文件              │
│   (模型/数据/方法/超参/硬件)          │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│          Axolotl Core               │
├──────────┬──────────┬───────────────┤
│ 数据处理  │ 训练引擎  │  分布式调度    │
│ - 多格式  │ - SFT    │  - DeepSpeed  │
│ - 打包    │ - DPO    │  - FSDP       │
│ - 验证集  │ - RLHF   │  - 多节点     │
│          │ - pretrain│              │
└──────────┴──────────┴───────────────┘
               │
┌──────────────▼──────────────────────┐
│        HuggingFace 生态             │
│  Transformers / PEFT / TRL / Datasets│
└─────────────────────────────────────┘
```

## 支持的训练方法

| 方法 | 类型 | 说明 |
|------|------|------|
| [[lora\|LoRA]] | PEFT | 低秩适配 |
| [[qlora\|QLoRA]] | PEFT+量化 | 4-bit量化 + LoRA |
| Full Fine-tune | 全量 | 全参数微调 |
| Freeze | 部分冻结 | 冻结部分层 |
| [[dpo\|DPO]] | 对齐 | 直接偏好优化 |
| [[rlhf\|RLHF]]/PPO | 对齐 | 人类反馈强化学习 |
| [[grpo\|GRPO]] | 对齐 | 组相对策略优化 |
| [[kto\|KTO]] | 对齐 | Kahneman-Tversky优化 |
| ORPO | 对齐 | 无参考模型的偏好优化 |

## 支持的模型

| 模型系列 | 具体模型 |
|----------|----------|
| Llama | Llama 2, Llama 3, Llama 3.1 |
| Mistral | Mistral-7B, Mixtral-8x7B |
| Qwen | Qwen2, Qwen2.5 |
| Yi | Yi-6B, Yi-34B |
| Gemma | Gemma-2B, Gemma-7B |
| DeepSeek | DeepSeek-V2, DeepSeek-R1 |
| Phi | Phi-3 |
| Code | CodeLlama, StarCoder2 |

## 关键特性

### 多GPU训练
```yaml
# deepspeed 配置示例
deepspeed: deepspeed_configs/zero3_bf16.json
# 自动处理：
# - 梯度分片
# - 优化器状态分片
# - 参数分片
# - 激活重计算
```

### 灵活的数据格式
```yaml
# 支持 JSONL, Parquet, HuggingFace 数据集
datasets:
  - path: mhenrichsen/alpaca_2k_test
    type: alpaca
  - path: local_data.jsonl
    type: sharegpt
```

### QLoRA 配置示例
```yaml
base_model: meta-llama/Meta-Llama-3-8B
load_in_4bit: true
adapter: qlora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

datasets:
  - path: dataset.jsonl
    type: alpaca

sequence_len: 2048
bf16: true
gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 3
learning_rate: 0.0002
```

## 与其他框架对比

| 特性 | Axolotl | [[llama-factory\|LLaMA Factory]] | [[trl\|TRL]] | [[unsloth\|Unsloth]] |
|------|---------|---------|-----|---------|
| 配置方式 | YAML | YAML + Web UI | Python 代码 | Python 代码 |
| 学习曲线 | 中等 | 低 | 高 | 低 |
| DeepSpeed 集成 | **最佳** | 支持 | 部分 | 不支持 |
| 多节点训练 | ✅ | ✅ | ❌ | ❌ |
| 方法覆盖 | **最全** | 广 | 深 | 窄 |
| Web UI | ❌ | ✅ | ❌ | ❌ |
| 代码干预 | 有限 | 有限 | **完全自由** | 有限 |

## 适用场景

### 推荐 Axolotl 的情况
- 需要 DeepSpeed 多卡/多节点分布式训练
- 需要组合多种训练方法（SFT → DPO → KTO）
- 追求配置即代码的可复现性
- 团队协作需要标准化训练流程

### 可选其他框架
- **[[llama-factory|LLaMA Factory]]**: 快速实验，Web UI
- **[[trl|TRL]]**: 研究用途，需深度定制
- **[[unsloth|Unsloth]]**: 单卡极致速度

## 相关资源

- GitHub: https://github.com/OpenAccess-AI-Collective/axolotl
- 文档: https://axolotl-ai-cloud.github.io/axolotl/
- 示例配置: https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples

## 相关内容

- [[llama-factory|LLaMA Factory]] — 另一主流微调框架
- [[trl|TRL]] — HuggingFace RL训练库
- [[unsloth|Unsloth]] — 极速训练框架
- [[lora|LoRA]] — 核心PEFT方法
- [[dpo|DPO]] — 直接偏好优化
- [[grpo|GRPO]] — 组相对策略优化
