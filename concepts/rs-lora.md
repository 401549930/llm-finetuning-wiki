---
title: RSLoRA: Rank-Stabilized LoRA
created: 2026-04-23
updated: 2026-04-23
type: method
tags: [lora, peft, rs-lora, fine-tuning]
sources: []
paper: "Rank-Stabilized LoRA: Scaling law and Convergence of LoRA"
arxiv_id: "2312.03732"
---

# RSLoRA: Rank-Stabilized LoRA

## 一句话总结
修正 LoRA 缩放因子，使高秩训练时效果不再退化，秩越高效果越好。

## 核心问题

标准 [[lora|LoRA]] 使用缩放因子 $\frac{\alpha}{r}$，这导致高秩时效果反而下降：

```
效果
  │     RSLoRA ────────
  │    ╱
  │   ╱
  │  ╱ LoRA ──── 效果下降！
  │ ╱  ╲
  │╱    ╲
  │______╲________ 秩 r
         ↑
     高秩时 LoRA 退化
```

### 原因分析

LoRA 输出 $h = W_0 x + \frac{\alpha}{r} BAx$

当秩 $r$ 增大时：
- $BA$ 的范数通常与 $\sqrt{r}$ 成正比增长
- $\frac{1}{r}$ 的衰减速度比 $\sqrt{r}$ 的增长更快
- 导致高秩时 LoRA 贡献变小，学习不充分

## RSLoRA 修正

将缩放因子从 $\frac{\alpha}{r}$ 改为 $\frac{\alpha}{\sqrt{r}}$：

$$h = W_0 x + \frac{\alpha}{\sqrt{r}} BAx$$

### 理论依据

根据缩放定律 (Scaling Law)：
- LoRA 更新的范数 $\|BA\| \propto \sqrt{r}$
- 缩放因子应与之匹配：$\frac{1}{\sqrt{r}}$ 使总贡献与秩无关
- 这样无论秩高低，学习率调优都是一致的

### 对比

| 方面 | LoRA | RSLoRA |
|------|------|--------|
| 缩放因子 | $\frac{\alpha}{r}$ | $\frac{\alpha}{\sqrt{r}}$ |
| 高秩效果 | 退化 | **持续提升** |
| 秩可扩展性 | 受限 | **无上限** |
| 向后兼容 | - | 兼容 (仅改缩放) |

## 实验结果

### 效果随秩变化

| 秩 r | LoRA (α/r) | RSLoRA (α/√r) |
|------|------------|---------------|
| 8 | 74.2 | 74.5 |
| 16 | 75.1 | 76.2 |
| 32 | 74.8 | **77.1** |
| 64 | 74.3 | **78.4** |
| 128 | 73.9 | **79.2** |

RSLoRA 在高秩时效果持续提升，LoRA 反而下降。

### 与全量微调对比

| 方法 | 秩 | MMLU | HumanEval |
|------|-----|------|-----------|
| Full | - | 65.2 | 42.1 |
| LoRA (r=64) | 64 | 62.8 | 38.5 |
| RSLoRA (r=64) | 64 | **64.5** | **41.3** |
| RSLoRA (r=128) | 128 | **65.0** | **41.8** |

## 实现示例

### HuggingFace PEFT
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=64,
    lora_alpha=64,
    use_rslora=True,  # 启用 RSLoRA
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, config)
```

### 手动实现
```python
class RSLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=64, alpha=64):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.W.weight.requires_grad = False

        self.A = nn.Parameter(torch.randn(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        # 关键区别: sqrt(r) 而非 r
        self.scaling = alpha / (r ** 0.5)

    def forward(self, x):
        return self.W(x) + (x @ self.A.T @ self.B.T) * self.scaling
```

## 配置建议

| 参数 | LoRA 建议 | RSLoRA 建议 |
|------|-----------|-------------|
| r | 8-32 | 16-128 (可更高) |
| alpha | 16-32 | 16-64 |
| 学习率 | 1e-4 | 1e-4 (无需改) |

RSLoRA 允许使用更高的秩而不担心效果退化，在需要更强表达力时特别有用。

## 与其他 LoRA 变体关系

| 变体 | 改进点 | 与 RSLoRA 兼容 |
|------|--------|---------------|
| [[qlora\|QLoRA]] | 量化 | ✅ |
| [[dora\|DoRA]] | 权重分解 | ✅ |
| RSLoRA | 缩放因子 | - |
| LoRA+ | 初始化改进 | ✅ |

可以组合使用：RSLoRA + QLoRA + DoRA

## 优缺点

| 优点 | 缺点 |
|------|------|
| 高秩效果持续提升 | 低秩时差异不大 |
| 修改量极小 | 需要框架支持 |
| 向后兼容 | 较新，实践较少 |
| 理论支撑强 | 高秩训练时间更长 |

## 相关论文

- Rank-Stabilized LoRA (Kalajdzievski, 2023)
- LoRA: Low-Rank Adaptation (Hu et al., 2021)
- LoRA+: Low-Rank Adaptations with Multiple Scaling Factors (2024)

## 相关内容

- [[lora|LoRA]] — 基础方法
- [[qlora|QLoRA]] — 量化 LoRA
- [[dora|DoRA]] — 权重分解 LoRA
