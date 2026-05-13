---
title: DoRA: Weight-Decomposed Low-Rank Adaptation
created: 2026-04-23
updated: 2026-04-23
type: method
tags: [dora, lora, peft, fine-tuning]
sources: []
paper: "DoRA: Weight-Decomposed Low-Rank Adaptation"
arxiv_id: "2402.09353"
---

# DoRA: Weight-Decomposed Low-Rank Adaptation

## 一句话总结
将权重分解为幅度和方向，分别用不同方式微调，效果逼近全量微调同时保持 LoRA 的参数效率。

## 核心原理

DoRA 的核心洞察：预训练权重 $W$ 可以分解为幅度向量 $m$ 和方向矩阵 $V$：

$$W = m \cdot V$$

其中 $m \in \mathbb{R}^{1 \times k}$ 是行向量，$V$ 是归一化的方向矩阵。

### 与 LoRA 的区别

| 方面 | [[lora\|LoRA]] | DoRA |
|------|---------------|------|
| 权重更新 | $\Delta W = BA$ | $\Delta W$ 分别作用于 m 和 V |
| 可训练参数 | BA 矩阵 | m 向量 + V 的 LoRA |
| 幅度调整 | 隐式 | **显式分离** |
| 方向调整 | 隐式 | **显式分离** |

### 权重分解

```
原始权重 W (d × k)
        │
        ▼
┌───────────────────────┐
│  W = m · V           │
│  m: 幅度向量 (1 × k)  │
│  V: 方向矩阵 (d × k)  │
│  V 已归一化          │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  微调时:              │
│  m → m + Δm (直接训练) │
│  V → V + ΔV (LoRA)    │
└───────────────────────┘
```

### 参数效率

对于 $W \in \mathbb{R}^{d \times k}$，秩为 $r$：
- LoRA 参数：$d \times r + r \times k$
- DoRA 参数：$k + d \times r + r \times k$（仅多 $k$ 个参数）

幅度向量 $m$ 参数量可忽略，但分离带来的收益显著。

## 算法细节

### 前向传播

```python
def dora_forward(W, m, V, lora_A, lora_B, x):
    # W = m · V (预分解)
    # 方向更新: ΔV = lora_B @ lora_A
    # 幅度更新: Δm

    # 方向更新 (类似 LoRA)
    delta_V = lora_B @ lora_A
    V_new = V + delta_V

    # 幅度更新
    m_new = m + delta_m

    # 组合
    W_new = m_new * V_new  # 广播乘法
    output = x @ W_new.T
    return output
```

### 初始化

```python
# 从预训练权重初始化
m = W.norm(dim=0, keepdim=True)  # 幅度
V = W / m                         # 归一化方向

# LoRA 零初始化
lora_B = torch.zeros(d, r)
lora_A = torch.randn(r, k) * 0.01
delta_m = torch.zeros(1, k)
```

## 性能对比

### 准确率 (GLUE 基准)

| 方法 | 参数量 | RTE | CoLA | SST-2 | MRPC | 平均 |
|------|--------|-----|------|-------|------|------|
| Full | 100% | 86.6 | 86.8 | 94.8 | 90.2 | **89.6** |
| LoRA | 0.2% | 85.1 | 84.4 | 94.0 | 88.2 | 87.9 |
| DoRA | 0.2% | **86.2** | **86.5** | **94.5** | **89.6** | **89.2** |

### 指令微调 (AlpacaEval 2.0)

| 方法 | 参数量 | Win Rate |
|------|--------|----------|
| Full Fine-tune | 100% | 28.1% |
| LoRA (r=64) | 1.1% | 24.2% |
| DoRA (r=64) | 1.1% | **27.4%** |

DoRA 几乎达到全量微调效果！

## 实现示例

### HuggingFace PEFT
```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# DoRA 配置 (PEFT 已支持)
config = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_dora=True,  # 启用 DoRA
)

model = get_peft_model(model, config)
```

### Axolotl 配置
```yaml
adapter: lora
lora_r: 64
lora_alpha: 64
use_dora: true  # 启用 DoRA
```

## 为什么 DoRA 更好？

### 学习曲线差异

```
学习进度
  │
  │    DoRA ────────
  │   ╱
  │  ╱
  │ ╱ LoRA
  │╱
  │________________ 训练步数

DoRA 学习更快，收敛更好
```

### 原因分析

1. **更新的可解释性**：幅度和方向解耦，各司其职
2. **幅度直接控制**：LoRA 中幅度变化隐式，需多步学习
3. **方向更新更纯净**：不受幅度干扰
4. **减少学习冲突**：两个目标分离优化

## 与其他 PEFT 方法对比

| 方法 | 参数量 | 效果 | 训练稳定性 |
|------|--------|------|------------|
| Full | 100% | 最佳 | 中 |
| [[lora|LoRA]] | 0.1-1% | 好 | **最高** |
| DoRA | 0.1-1% | **接近最佳** | 高 |
| [[adapter|Adapter]] | 1-5% | 好 | 高 |
| [[prefix-tuning|Prefix-Tuning]] | <0.1% | 中 | 中 |

## 优缺点

| 优点 | 缺点 |
|------|------|
| 效果逼近全量微调 | 比 LoRA 多少量参数 |
| 参数效率与 LoRA 相当 | 实现复杂度略高 |
| 训练稳定 | 融合权重时需额外处理 |
| 收敛更快 | 较新，生态支持尚在完善 |

## 适用场景

- 需要 LoRA 的参数效率但追求更高效果
- 指令微调任务
- 领域适配任务
- 替代全量微调

## 相关论文

- DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024)
- LoRA: Low-Rank Adaptation (Hu et al., 2021)

## 相关内容

- [[lora|LoRA]] — DoRA 的基础方法
- [[qlora|QLoRA]] — 可与 DoRA 组合
- [[adapter|Adapter]] — 另一种 PEFT 方法
- [[rs-lora|RSLoRA]] — LoRA 的秩稳定改进
