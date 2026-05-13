---
title: KTO: Kahneman-Tversky Optimization
created: 2026-04-23
updated: 2026-04-23
type: method
tags: [kto, alignment, rlhf, dpo, preference-optimization]
sources: []
paper: "KTO: Model Alignment as Prospect Theoretic Optimization"
arxiv_id: "2402.01306"
---

# KTO: Kahneman-Tversky Optimization

## 一句话总结
基于前景理论的人类偏好模型，仅需"好/坏"二元标注即可优化，无需成对偏好数据。

## 核心原理

KTO 的核心洞察：人类对损失的敏感度约为收益的2倍 (Kahneman-Tversky 前景理论)。将这一认知偏差建模进损失函数，实现更符合人类决策的对齐。

### 与 DPO 的关键区别

| 方面 | [[dpo\|DPO]] | KTO |
|------|-------------|-----|
| 数据要求 | 成对偏好 (chosen vs rejected) | **单点标注** (好/坏) |
| 数据获取 | 困难 (需配对) | **容易** |
| 理论基础 | Bradley-Terry 模型 | **前景理论** |
| 损失对称性 | 对称 | **非对称** (损失权重更大) |

### 损失函数

$$\mathcal{L}_{KTO} = \lambda_w \cdot \mathbb{E}_{x,y_w \sim D_w}[\ell(v(x,y_w))] + \lambda_l \cdot \mathbb{E}_{x,y_l \sim D_l}[\ell(-v(x,y_l))]$$

其中价值函数：
$$v(x,y) = \beta \left(\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} - z_{ref}\right)$$

$z_{ref}$ 是参考模型的 KL 散度基线，确保正值代表收益、负值代表损失。

关键：$\lambda_l > \lambda_w$，损失权重更大，反映前景理论中的损失厌恶。

### 前景理论价值函数

```
价值 v
  │    ╱
  │   ╱  收益 (凹函数)
  │  ╱   敏感度递减
  │ ╱
──╳────────── 结果
  │╲
  │ ╲  损失 (凸函数)
  │  ╲  更陡峭
  │   ╲ 损失厌恶: λ_l > λ_w
```

## 实现示例

### TRL 中使用 KTO
```python
from trl import KTOTrainer, KTOConfig

config = KTOConfig(
    beta=0.1,                  # KL 约束系数
    loss_type="kto",           # KTO 损失
    desirable_weight=1.0,      # λ_w
    undesirable_weight=1.0,    # λ_l (可调大)
    per_device_train_batch_size=8,
)

trainer = KTOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=kto_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

### 数据格式
```json
// KTO 只需要单点标注
{"prompt": "解释量子力学", "completion": "量子力学是...", "label": true}   // 好的回答
{"prompt": "解释量子力学", "completion": "我不确定...", "label": false}  // 差的回答
```

对比 DPO 需要成对数据：
```json
{"prompt": "解释量子力学", "chosen": "好的回答", "rejected": "差的回答"}
```

## 数据效率

| 数据类型 | DPO | KTO |
|----------|-----|-----|
| 成对偏好 | ✅ 必需 | ❌ 不需要 |
| 二元标注 | ❌ 不支持 | ✅ 充分 |
| 数据获取成本 | 高 | **低** |
| 数据利用率 | 1对/样本 | **1条/样本** |

实际测试中，KTO 在数据量较少时效果甚至优于 DPO，因为：
1. 更多可用数据 (不需要配对)
2. 损失函数更符合人类认知
3. 对极端样本更鲁棒

## 实验结果

| 方法 | 数据类型 | AlpacaEval 2.0 | MT-Bench |
|------|----------|----------------|----------|
| SFT | 指令 | 5.0 | 6.5 |
| DPO | 成对偏好 | 12.2 | 7.3 |
| KTO | 二元标注 | **13.5** | **7.5** |
| RLHF/PPO | 成对偏好 + RM | 14.1 | 7.8 |

## 优缺点

| 优点 | 缺点 |
|------|------|
| 数据获取成本低 | 理论较新，实践案例少 |
| 单点标注即可 | 对好/坏比例敏感 |
| 损失厌恶建模更合理 | 可能过度惩罚 |
| 极端样本更鲁棒 | 需要合理的 λ_w/λ_l 配比 |

## 适用场景

- 只有二元标注数据（赞/踩）
- 数据标注预算有限
- 需要快速对齐
- 不便获取成对偏好数据

## 相关论文

- KTO: Model Alignment as Prospect Theoretic Optimization (Ethayarajh et al., 2024)
- Prospect Theory: An Analysis of Decision under Risk (Kahneman & Tversky, 1979)

## 相关内容

- [[dpo|DPO]] — 成对偏好优化
- [[rlhf|RLHF]] — 人类反馈强化学习
- [[grpo|GRPO]] — 组相对策略优化
- [[trl|TRL]] — 支持 KTO 训练
