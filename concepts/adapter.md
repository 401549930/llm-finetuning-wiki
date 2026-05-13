---
title: Adapter: Parameter-Efficient Transfer Learning
created: 2026-04-23
updated: 2026-04-23
type: method
tags: [adapter, peft, transfer-learning]
sources: [raw/papers/arxiv-1902.00751.md]
arxiv_id: "1902.00751"
---

# Adapter: Parameter-Efficient Transfer Learning

## 一句话总结
在 Transformer 层中插入小型瓶颈模块，仅训练这些模块实现高效迁移。

## 核心原理

Adapter 在每层 Transformer 中插入两个小型全连接网络：

```
输入 → Self-Attention → Adapter → Add & Norm → Feed-Forward → Adapter → Add & Norm → 输出
                              ↑                              ↑
                          仅训练这里                      仅训练这里
```

### Adapter 模块结构

```
d 维输入
    ↓
投影到 m 维 (d × m 矩阵, m << d)
    ↓
非线性激活 (ReLU/GELU)
    ↓
投影回 d 维 (m × d 矩阵)
    ↓
残差连接
```

参数量：$2 \times d \times m$（每个 Adapter）

## 实现要点

```python
class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck=64):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck)
        self.up_proj = nn.Linear(bottleneck, d_model)
        self.activation = nn.ReLU()
        # 初始化接近恒等映射
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return x + residual  # 残差连接
```

## 关键设计

### 瓶颈维度
- 通常 $m = d/10$ 或更小
- BERT-base (d=768)：$m=64$，每层仅 98,304 参数
- 相比全量微调：减少 1000x 参数

### 插入位置
- 论文建议：每个 Transformer 块插入 2 个 Adapter
- 位置：Self-Attention 后 + Feed-Forward 后

### 初始化策略
- 下投影：随机初始化
- 上投影：零初始化
- 效果：初始状态接近恒等映射

## 实验结果

在 GLUE 基准上：
| 方法 | 参数量 | GLUE 平均分 |
|------|--------|-------------|
| 全量微调 | 100% | 84.3 |
| Adapter | 3.6% | 83.9 |
| Adapter (小) | 0.5% | 82.1 |

## 优缺点

| 优点 | 缺点 |
|------|------|
| 参数效率极高 | 增加推理延迟 |
| 易于扩展新任务 | 瓶颈可能限制表达能力 |
| 保持原模型冻结 | 需要修改模型结构 |
| 多任务友好 | 实现相对复杂 |

## 与 LoRA 对比

| 方面 | Adapter | LoRA |
|------|---------|------|
| 参数位置 | 层间插入 | 权重旁路 |
| 推理延迟 | 有增加 | 无（可合并） |
| 实现复杂度 | 需改结构 | 即插即用 |
| 参数效率 | 很高 | 高 |
| 效果 | 好 | 略好 |

## 适用场景

- 多任务模型共享
- 参数极度受限场景
- 需要保留多个任务能力
- 研究和新方法探索

## 相关方法

- [[lora|LoRA]]：另一种 PEFT 方法
- [[prefix-tuning|Prefix-Tuning]]：前缀微调
- [[p-tuning|P-Tuning]]：连续提示微调
- [[qlora|QLoRA]]：量化 + LoRA

## 主要论文

- [[raw/papers/arxiv-1902.00751|Parameter-Efficient Transfer Learning for NLP]] (Houlsby et al., 2019)

## 参考文献

- AdapterHub: https://adapterhub.ml/
- Hugging Face Adapters: https://docs.adapterhub.ml/
