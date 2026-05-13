---
title: Prefix-Tuning: Optimizing Continuous Prompts
created: 2026-04-23
updated: 2026-04-23
type: method
tags: [prefix-tuning, peft, prompt-tuning]
sources: [raw/papers/arxiv-2101.00190.md]
arxiv_id: "2101.00190"
---

# Prefix-Tuning: Optimizing Continuous Prompts

## 一句话总结
在每层 Transformer 前添加可训练的前缀向量，仅训练这些前缀实现高效微调。

## 核心原理

Prefix-Tuning 在每一层的注意力计算中插入"虚拟 token"：

```
原始注意力:
Q = W_q · x
K = W_k · x
V = W_v · x
Attention = softmax(Q·K^T / √d) · V

Prefix-Tuning:
Q = W_q · x
K = [P_k; W_k · x]    ← 前缀拼接到 Key
V = [P_v; W_v · x]    ← 前缀拼接到 Value
Attention = softmax(Q·K^T / √d) · V
```

### 关键设计
- 前缀向量 $(P_k, P_v)$ 是可训练参数
- 每层都有独立的前缀
- 模型权重完全冻结

## 实现要点

```python
class PrefixEncoder(nn.Module):
    def __init__(self, num_layers, num_heads, prefix_len, hidden_size):
        super().__init__()
        self.prefix_len = prefix_len
        
        # 直接参数化（小模型）
        # 或用 MLP 重新参数化（大模型，训练更稳定）
        self.prefix_embeddings = nn.Parameter(
            torch.randn(num_layers * 2 * prefix_len, num_heads, hidden_size // num_heads)
        )
        
        # 重新参数化（推荐）
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_layers * 2 * prefix_len * hidden_size)
        )
    
    def forward(self):
        # 返回每层的前缀 (key, value)
        return self.mlp(self.prefix_embeddings).view(
            self.num_layers, 2, self.prefix_len, self.num_heads, self.head_dim
        )
```

## 参数量对比

GPT-2 Medium (355M 参数)：
| 方法 | 参数量 | 占比 |
|------|--------|------|
| 全量微调 | 355M | 100% |
| Prefix-Tuning (prefix_len=5) | 360K | 0.1% |
| Prefix-Tuning (prefix_len=10) | 720K | 0.2% |

## 实验结果

### 表格转文本（E2E 数据集）
| 方法 | BLEU | NIST |
|------|------|------|
| 全量微调 | 46.2 | 8.62 |
| Prefix-Tuning | 45.5 | 8.49 |

### 摘要任务（XSum 数据集）
| 方法 | RG-1 | RG-2 | RG-L |
|------|------|------|------|
| 全量微调 | 43.3 | 20.0 | 35.2 |
| Prefix-Tuning | 43.2 | 20.1 | 35.3 |

## 低数据优势

在数据量少时，Prefix-Tuning 往往**优于全量微调**：

| 数据比例 | 全量微调 | Prefix-Tuning |
|----------|----------|---------------|
| 100% | 46.2 | 45.5 |
| 10% | 39.5 | 42.0 |
| 1% | 30.8 | 36.9 |

## 优缺点

| 优点 | 缺点 |
|------|------|
| 参数量极小 (0.1%) | 批次大小受限于前缀长度 |
| 低数据场景优势明显 | 长前缀可能影响生成 |
| 多任务存储友好 | 实现相对复杂 |
| 外推能力好 | 推理时有额外开销 |

## 与其他 PEFT 对比

| 方法 | 参数位置 | 参数量 | 推理开销 |
|------|----------|--------|----------|
| Prefix-Tuning | 每层注意力 | 0.1% | 有 |
| P-Tuning | 输入层 | 更少 | 无 |
| LoRA | 权重旁路 | 0.1-1% | 无（可合并） |
| Adapter | 层间模块 | 1-5% | 有 |

## 适用场景

- 低数据量场景
- 多任务模型共享
- 需要外推到未见主题
- 参数极度受限

## 相关方法

- [[p-tuning|P-Tuning]]：输入层连续提示
- [[lora|LoRA]]：低秩适配
- [[adapter|Adapter]]：层间插入模块
- [[prompt-tuning|Prompt Tuning]]：仅输入层

## 主要论文

- [[raw/papers/arxiv-2101.00190|Prefix-Tuning: Optimizing Continuous Prompts for Generation]] (Li & Liang, 2021)

## 参考文献

- GitHub: https://github.com/XiangLi1999/PrefixTuning
