---
title: DeepSpeed: 深度学习分布式训练框架
created: 2026-04-23
updated: 2026-04-23
type: entity
tags: [deepspeed, distributed-training, fsdp, zero-offload, zeRO]
github: https://github.com/microsoft/DeepSpeed
stars: 35000+
developer: Microsoft
---

# DeepSpeed

## 概述

DeepSpeed 是微软开发的开源深度学习优化库，专注于大规模模型的高效训练。其核心贡献是 ZeRO (Zero Redundancy Optimizer) 技术，通过智能分片消除数据并行中的冗余。

**核心价值**：
- 训练超大模型 (万亿参数级别)
- 内存效率提升 10x
- 训练速度提升 3-5x
- 支持各种硬件配置

## ZeRO 技术

### ZeRO 优化层级

| Level | 分片内容 | 内存节省 | 通信开销 |
|-------|----------|----------|----------|
| ZeRO-1 | 优化器状态 | **4x** | 低 |
| ZeRO-2 | + 梯度 | **8x** | 中 |
| ZeRO-3 | + 模型参数 | **N倍** (N=GPU数) | 高 |
| ZeRO-Offload | + CPU卸载 | **极大** | 较高 |

### ZeRO-3 内存分布
```
传统数据并行:
┌──────────────────────────────────────────────┐
│    GPU 0              GPU 1              GPU N   │
│  ┌─────────┐       ┌─────────┐       ┌─────────┐ │
│  │ 全量参数 │       │ 全量参数 │       │ 全量参数 │ │
│  │ 优化状态 │       │ 优化状态 │       │ 优化状态 │ │
│  │ 梯度    │       │ 梯度    │       │ 梯度    │ │
│  └─────────┘       └─────────┘       └─────────┘ │
│  (冗余存储)                                    │
└──────────────────────────────────────────────┘

ZeRO-3 并行:
┌──────────────────────────────────────────────┐
│ GPU 0       GPU 1       GPU 2       GPU N-1    │
│ ┌────┐     ┌────┐     ┌────┐     ┌────┐       │
│ │1/N │     │1/N │     │1/N │     │1/N │       │
│ │参数│     │参数│     │参数│     │参数│       │
│ │优化│     │优化│     │优化│     │优化│       │
│ │梯度│     │梯度│     │梯度│     │梯度│       │
│ └────┘     └────┘     └────┘     └────┘       │
│ (各存 1/N，需要时 all-gather)                 │
└──────────────────────────────────────────────┘
```

## 核心功能

### 1. 模型并行
- Tensor 并行 (配合 Megatron)
- Pipeline 并行
- 混合并行

### 2. 内存优化
- ZeRO-Offload: CPU 内存卸载
- ZeRO-Infinity: NVMe 卸载
- Activation Checkpointing: 激活重计算

### 3. 训练加速
- Gradient Compression: 梯度压缩
- Communication Overlap: 通信计算重叠
- Mixed Precision: 混合精度

## 配置示例

### ZeRO-2 配置
```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "reduce_scatter": true,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "bf16": { "enabled": true },
  "gradient_accumulation_steps": "auto"
}
```

### ZeRO-3 配置
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu"
    },
    "offload_param": {
      "device": "cpu"
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto"
  },
  "bf16": { "enabled": true }
}
```

### Axolotl 中使用
```yaml
deepspeed: deepspeed_configs/zero3_bf16.json
# 或
deepspeed: deepspeed_configs/zero2_offload.json

# 自动配置
bf16: true
```

## 显存计算

对于 Llama-65B 模型 (FP16):

| 配置 | 显存需求 |
|------|----------|
| 标准 DP | 650GB+ |
| ZeRO-1 | ~160GB |
| ZeRO-2 | ~80GB |
| ZeRO-3 (8 GPU) | ~20GB |
| ZeRO-3 + Offload | **<10GB** |

## 与 FSDP 关系

| 特性 | DeepSpeed | PyTorch FSDP |
|------|-----------|--------------|
| 开发者 | Microsoft | Meta/PyTorch |
| ZeRO-3 | ✅ | ✅ |
| CPU Offload | ✅ | ✅ |
| NVMe Offload | ✅ | ❌ |
| 配置复杂度 | JSON 配置 | Python 代码 |
| HuggingFace | 深度集成 | 原生支持 |

## 相关资源

- GitHub: https://github.com/microsoft/DeepSpeed
- 文档: https://www.deepspeed.ai
- 教程: https://www.deepspeed.ai/tutorials

## 相关内容

- [[axolotl|Axolotl]] — 深度集成 DeepSpeed
- [[llama-factory|LLaMA Factory]] — 支持 DeepSpeed
- [[trl|TRL]] — 部分支持
- [[lora|LoRA]] — 常与 DeepSpeed 组合
- [[qlora|QLoRA]] — 量化替代方案
