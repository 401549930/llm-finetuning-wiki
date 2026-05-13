---
title: LoRA: Low-Rank Adaptation
created: 2026-04-23
updated: 2026-04-23
type: method
tags: [lora, peft, fine-tuning]
sources: [raw/papers/arxiv-2106.09685.md]
arxiv_id: "2106.09685"
---

# LoRA: Low-Rank Adaptation

## 一句话总结
冻结预训练权重，通过低秩分解矩阵注入可训练参数，实现高效微调。

## 核心原理

LoRA 的核心假设：模型适配过程中的权重更新是低秩的。

对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 将更新 $\Delta W$ 分解为：

$$W = W_0 + \Delta W = W_0 + BA$$

其中：
- $B \in \mathbb{R}^{d \times r}$（可训练）
- $A \in \mathbb{R}^{r \times k}$（可训练）
- 秩 $r \ll \min(d, k)$，通常 $r=8, 16, 32$

### 关键设计
1. **冻结原始权重**：$W_0$ 保持不变，只训练 $A$ 和 $B$
2. **零初始化**：$B$ 初始化为 0，$A$ 随机初始化，确保初始状态 $\Delta W = 0$
3. **缩放因子**：输出乘以 $\frac{\alpha}{r}$，$\alpha$ 是超参数

## 实现要点

```python
# PyTorch 伪代码
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.W.weight.requires_grad = False  # 冻结
        
        self.A = nn.Parameter(torch.randn(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r
    
    def forward(self, x):
        return self.W(x) + (x @ self.A.T @ self.B.T) * self.scaling
```

### 推荐配置
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| r | 8-64 | 秩，越大效果越好但参数越多 |
| alpha | 16-32 | 缩放因子，通常设为 2r |
| target_modules | q_proj, v_proj | 默认只微调注意力层 |

## 优缺点

| 优点 | 缺点 |
|------|------|
| 参数量减少 10,000x | 部分任务效果略低于全量微调 |
| 显存占用减少 3x | 秩选择需要调参 |
| 无额外推理延迟 | 超参数敏感 |
| 可合并权重，部署不变 | - |

## 参数量对比

以 GPT-3 175B 为例：
- 全量微调：175B 参数
- LoRA (r=8)：约 17M 参数（减少 10,000 倍）

## 适用场景

- 大模型单卡微调（资源受限）
- 多任务部署（每个任务一个小 adapter）
- 快速实验迭代
- 需要保留原始模型能力

## 相关方法

- [[qlora|QLoRA]]：结合量化的 LoRA
- [[adapter|Adapter]]：另一种参数高效微调方法
- [[prefix-tuning|Prefix-Tuning]]：前缀微调
- [[p-tuning|P-Tuning]]：连续提示微调

## 主要论文

- [[raw/papers/arxiv-2106.09685|LoRA: Low-Rank Adaptation of Large Language Models]] (Hu et al., 2021)

## 参考文献

- GitHub: https://github.com/microsoft/LoRA
- Hugging Face PEFT: https://huggingface.co/docs/peft/conceptual_guides/lora
