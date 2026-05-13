---
title: PEFT 方法对比
created: 2026-04-23
updated: 2026-04-23
type: comparison
tags: [comparison, peft, lora, adapter, prefix-tuning]
sources: []
---

# PEFT 方法对比

## 对比概览

| 方法 | 参数占比 | 推理延迟 | 训练稳定性 | 实现复杂度 |
|------|----------|----------|------------|------------|
| [[lora|LoRA]] | 0.1-1% | 无 | 高 | 低 |
| [[qlora|QLoRA]] | 0.1-1% | 无 | 高 | 中 |
| [[adapter|Adapter]] | 1-5% | 有 | 高 | 中 |
| [[prefix-tuning|Prefix-Tuning]] | 0.1% | 有 | 中 | 高 |
| [[p-tuning|P-Tuning]] | <0.1% | 无 | 高 | 低 |

## 详细对比

### 参数效率

| 方法 | 典型参数量 (7B模型) | 存储开销 |
|------|---------------------|----------|
| 全量微调 | 7B | 14GB (fp16) |
| LoRA (r=8) | ~4M | 8MB |
| QLoRA (r=8) | ~4M | 8MB |
| Adapter (bottleneck=64) | ~20M | 40MB |
| Prefix-Tuning (len=10) | ~3M | 6MB |
| P-Tuning (len=20) | ~1M | 2MB |

### 显存需求 (7B 模型)

| 方法 | 最小显存 | 推荐显存 |
|------|----------|----------|
| 全量微调 (fp16) | 28GB | 40GB+ |
| LoRA (fp16) | 16GB | 24GB |
| QLoRA (4-bit) | 6GB | 12GB |
| Adapter | 18GB | 24GB |
| Prefix-Tuning | 14GB | 20GB |

### 训练效果

| 方法 | 全数据效果 | 少数据效果 | 外推能力 |
|------|------------|------------|----------|
| 全量微调 | 最佳 | 差 | 中 |
| LoRA | 接近最佳 | 中 | 中 |
| QLoRA | 接近最佳 | 中 | 中 |
| Adapter | 接近最佳 | 中 | 中 |
| Prefix-Tuning | 接近最佳 | **最佳** | **最佳** |
| P-Tuning | 中 | 好 | 好 |

### 实现难度

| 方法 | 代码修改量 | 调参难度 | 框架支持 |
|------|------------|----------|----------|
| LoRA | 低 | 低 | 广泛 |
| QLoRA | 中 | 中 | 广泛 |
| Adapter | 中 | 低 | 中等 |
| Prefix-Tuning | 高 | 中 | 较少 |
| P-Tuning | 低 | 低 | 中等 |

## 选择指南

### 按资源选择

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| 单卡消费级 GPU (≤12GB) | **QLoRA** | 显存最优 |
| 单卡高端 GPU (24GB) | LoRA | 性价比最优 |
| 多卡训练 | LoRA / Adapter | 并行效率高 |
| CPU 训练 | 不推荐 | 太慢 |

### 按任务选择

| 任务类型 | 推荐方法 | 理由 |
|----------|----------|------|
| 分类/标注 | P-Tuning | 简单有效 |
| 生成任务 | LoRA / Adapter | 效果稳定 |
| 低数据场景 | Prefix-Tuning | 外推能力强 |
| 多任务部署 | Adapter / LoRA | 参数独立 |
| 对齐/RLHF | LoRA + DPO | 组合最佳 |

### 按模型规模选择

| 模型大小 | 推荐方法 |
|----------|----------|
| ≤1B | 全量微调 或 LoRA |
| 1B-7B | LoRA |
| 7B-33B | LoRA / QLoRA |
| 33B-70B | **QLoRA** |
| 70B+ | QLoRA（必需量化） |

## 组合使用

### QLoRA + DPO
最流行的开源对齐方案：
```python
# 1. QLoRA 微调
model = load_4bit_model("Llama-2-70b")
model = add_lora(model, r=16)
train_sft(model, data)

# 2. DPO 对齐
train_dpo(model, preference_data)
```

### LoRA + 多任务
每个任务一个 LoRA adapter：
```
base_model (frozen)
    ├── task_A_lora
    ├── task_B_lora
    └── task_C_lora
```

## 实践建议

1. **从 LoRA 开始**：最简单、支持最广
2. **显存不够用 QLoRA**：几乎无代价
3. **少样本优先 Prefix-Tuning**：外推能力最强
4. **多任务用 Adapter**：任务隔离清晰
5. **避免 P-Tuning 做生成**：效果不稳定

## 相关内容

- [[lora|LoRA 详解]]
- [[qlora|QLoRA 详解]]
- [[adapter|Adapter 详解]]
- [[prefix-tuning|Prefix-Tuning 详解]]
- [[p-tuning|P-Tuning 详解]]
- [[dpo|DPO 详解]]
- [[rlhf|RLHF 详解]]
