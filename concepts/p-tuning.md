---
title: P-Tuning: Continuous Prompt Embeddings
created: 2026-04-23
updated: 2026-04-23
type: method
tags: [p-tuning, peft, prompt-tuning]
sources: [raw/papers/arxiv-2103.10385.md]
arxiv_id: "2103.10385"
---

# P-Tuning: Continuous Prompt Embeddings

## 一句话总结
用可训练的连续向量替代离散提示词，实现提示的自动优化。

## 核心原理

### 问题：离散提示不稳定
传统提示工程：
- "This movie is [MASK]" → 效果好
- "The movie is [MASK]" → 效果差

改变一个词，效果可能大幅下降。

### 解决方案：连续提示

将离散提示 token 替换为可训练的连续向量：

```
原始: [CLS] It was [MASK] . [SEP]
         ↓
P-Tuning: [CLS] [h1] [h2] [h3] [MASK] . [SEP]
                ↓    ↓    ↓
           可训练向量
```

这些 [h_i] 是连续嵌入，不对应任何词表中的词。

## 实现要点

```python
class PTuningEmbedding(nn.Module):
    def __init__(self, num_virtual_tokens, hidden_size):
        super().__init__()
        # 可训练的虚拟 token 嵌入
        self.virtual_embeddings = nn.Embedding(num_virtual_tokens, hidden_size)
        # 初始化
        nn.init.normal_(self.virtual_embeddings.weight, std=0.02)
    
    def forward(self, input_ids, word_embeddings):
        # 获取虚拟 token 嵌入
        virtual_tokens = torch.arange(self.num_virtual_tokens).to(input_ids.device)
        virtual_embeds = self.virtual_embeddings(virtual_tokens)
        
        # 获取实际 token 嵌入
        word_embeds = word_embeddings(input_ids)
        
        # 拼接
        return torch.cat([virtual_embeds, word_embeds], dim=1)
```

## 两种模式

### 1. Frozen LM
- 语言模型权重冻结
- 只训练提示嵌入
- 参数量极少

### 2. Tuned LM
- 语言模型权重也训练
- 提示嵌入 + LM 一起训练
- 效果更好

## 实验结果

在 LAMA 和 SuperGLUE 上：
| 方法 | LAMA | SuperGLUE |
|------|------|-----------|
| Manual Prompt | 25.7 | 65.1 |
| P-Tuning (Frozen) | 43.2 | 72.1 |
| P-Tuning (Tuned) | 52.3 | 78.5 |

## 关键发现

1. **稳定性大幅提升**：不同初始化效果接近
2. **效果显著提升**：超越人工设计提示
3. **适用于少样本**：低资源场景优势明显
4. **跨任务迁移**：提示可跨任务复用

## 优缺点

| 优点 | 缺点 |
|------|------|
| 无需人工设计提示 | 需要调整虚拟 token 数量 |
| 训练稳定 | 不如 LoRA 直观 |
| 参数效率高 | 对任务格式敏感 |
| 适用于少样本 | 长度受限 |

## 与 Prefix-Tuning 对比

| 方面 | P-Tuning | Prefix-Tuning |
|------|----------|---------------|
| 插入位置 | 输入层 | 每层 |
| 复杂度 | 简单 | 复杂 |
| 参数量 | 少 | 多 |
| 效果 | 好 | 更好 |
| 发表时间 | 2021.03 | 2021.01 |

## 适用场景

- 少样本学习
- 提示工程自动化
- 资源受限场景
- 需要稳定训练

## 相关方法

- [[prefix-tuning|Prefix-Tuning]]：每层添加前缀
- [[adapter|Adapter]]：层间插入模块
- [[lora|LoRA]]：低秩适配
- [[prompt-tuning|Prompt Tuning]]：简化版 P-Tuning

## 主要论文

- [[raw/papers/arxiv-2103.10385|GPT Understands, Too]] (Liu et al., 2021)

## 参考文献

- GitHub: https://github.com/THUDM/P-tuning
