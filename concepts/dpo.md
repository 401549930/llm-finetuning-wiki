---
title: DPO: Direct Preference Optimization
created: 2026-04-23
updated: 2026-04-23
type: method
tags: [dpo, alignment, rlhf]
sources: [raw/papers/arxiv-2305.18290.md]
arxiv_id: "2305.18290"
---

# DPO: Direct Preference Optimization

## 一句话总结
绕过奖励模型，直接用偏好数据优化语言模型，简化 RLHF 流程。

## 核心原理

### RLHF 的问题
传统 RLHF 需要：
1. 训练奖励模型
2. 用 PPO 优化策略
3. 复杂的超参数调优
4. 训练不稳定

### DPO 的突破

DPO 发现：**最优策略可以通过奖励函数的闭式解表示**

从 Bradley-Terry 偏好模型出发，推导出：

$$p^*(y \succ y' | x) = \frac{\exp(r^*(x, y))}{\exp(r^*(x, y)) + \exp(r^*(x, y'))}$$

反过来，最优奖励函数可以表示为：

$$r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**关键洞察**：不需要显式学习 $r^*$，可以直接学习 $\pi^*$！

## DPO 损失函数

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

其中：
- $y_w$：更优回答（winner）
- $y_l$：较差回答（loser）
- $\beta$：KL 散度约束强度
- $\pi_{ref}$：参考模型（通常是 SFT 模型）

## 实现要点

```python
import torch.nn.functional as F

def dpo_loss(policy, ref_policy, beta=0.1):
    """
    policy: 当前模型的 log_probs
    ref_policy: 参考模型的 log_probs
    """
    # 计算对数概率差异
    log_ratio_w = policy.log_probs_w - ref_policy.log_probs_w
    log_ratio_l = policy.log_probs_l - ref_policy.log_probs_l
    
    # DPO 损失
    logits = beta * (log_ratio_w - log_ratio_l)
    loss = -F.logsigmoid(logits).mean()
    
    return loss
```

## RLHF vs DPO 对比

| 方面 | RLHF (PPO) | DPO |
|------|------------|-----|
| 奖励模型 | 需要 | 不需要 |
| 训练复杂度 | 高 | 低 |
| 超参数 | 多 | 少（主要 β） |
| 稳定性 | 不稳定 | 稳定 |
| 显存占用 | 高 | 低 |
| 效果 | 好 | 相当或更好 |

## 优缺点

| 优点 | 缺点 |
|------|------|
| 无需奖励模型 | 需要大量偏好数据 |
| 训练稳定 | 可能不如 RLHF 灵活 |
| 计算高效 | 对 β 参数敏感 |
| 易于实现 | 难以处理在线学习 |

## 适用场景

- 中小规模模型对齐
- 资源受限的团队
- 快速实验迭代
- 偏好数据充足

## 推荐配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| β | 0.1-0.5 | KL 约束强度 |
| 学习率 | 1e-6 到 5e-7 | 比微调更小 |
| 批大小 | 64-128 | 偏好对数量 |

## 相关方法

- [[rlhf|RLHF]]：传统方法
- [[kto|KTO]]：无需成对偏好数据
- [[spo|SPO]]：简化版 DPO
- [[grpo|GRPO]]：改进的 RL 方法

## 主要论文

- [[raw/papers/arxiv-2305.18290|Direct Preference Optimization: Your Language Model is Secretly a Reward Model]] (Rafailov et al., 2023)

## 参考文献

- Hugging Face TRL: https://huggingface.co/docs/trl/main/en/dpo_trainer
- Stanford CRFM: https://crfm.stanford.edu/
