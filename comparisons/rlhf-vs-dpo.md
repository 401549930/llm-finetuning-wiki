---
title: RLHF vs DPO 对比
created: 2026-04-23
updated: 2026-04-23
type: comparison
tags: [comparison, rlhf, dpo, alignment]
sources: []
---

# RLHF vs DPO 对比

## 核心差异

| 方面 | [[rlhf|RLHF]] | [[dpo|DPO]] |
|------|---------------|-------------|
| 奖励模型 | 需要单独训练 | 不需要 |
| 优化算法 | PPO（强化学习） | 分类损失 |
| 训练复杂度 | 高 | 低 |
| 超参数数量 | 多（10+） | 少（主要 β） |
| 显存需求 | 高（需要 RM + Policy） | 中（仅 Policy） |
| 训练稳定性 | 不稳定 | 稳定 |

## 流程对比

### RLHF 流程
```
预训练模型
    ↓
① SFT 微调（有监督）
    ↓
② 训练奖励模型（人类排序数据）
    ↓
③ PPO 优化（强化学习）
    ↓
对齐后的模型
```

### DPO 流程
```
预训练模型
    ↓
① SFT 微调（有监督）
    ↓
② DPO 优化（直接用偏好数据）
    ↓
对齐后的模型
```

**关键区别**：DPO 跳过了奖励模型训练和 PPO 优化。

## 数学对比

### RLHF 目标函数
$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)} [r(x, y)] - \beta \mathbb{E}_{x \sim \mathcal{D}} [D_{KL}(\pi(\cdot|x) \| \pi_{ref}(\cdot|x))]$$

需要：
1. 先学习奖励函数 $r(x, y)$
2. 再用 PPO 优化策略 $\pi$

### DPO 目标函数
$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

直接优化策略，无需奖励模型。

## 效果对比

### 情感控制任务
| 方法 | 目标情感准确率 |
|------|----------------|
| PPO (RLHF) | 84.6% |
| DPO | **92.1%** |

### 摘要任务
| 方法 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|---------|---------|---------|
| PPO (RLHF) | 44.2 | 21.3 | 36.1 |
| DPO | 44.0 | 21.1 | 35.9 |

### 对话任务
| 方法 | 人类偏好率 |
|------|------------|
| PPO (RLHF) | 50% |
| DPO | 50% |

**结论**：DPO 在大多数任务上与 RLHF 效果相当，在部分任务上更好。

## 资源需求对比

### 显存 (7B 模型)
| 组件 | RLHF | DPO |
|------|------|-----|
| Policy 模型 | 14GB | 14GB |
| Reward 模型 | 4GB | - |
| Reference 模型 | 14GB | 14GB |
| **总计** | **32GB** | **28GB** |

### 训练时间
| 阶段 | RLHF | DPO |
|------|------|-----|
| SFT | 3h | 3h |
| 奖励模型 | 2h | - |
| PPO/DPO | 5h | 1h |
| **总计** | **10h** | **4h** |

## 超参数对比

### RLHF 关键超参数
| 参数 | 典型值 | 敏感度 |
|------|--------|--------|
| learning_rate | 1e-6 | 高 |
| ppo_clip_range | 0.2 | 中 |
| kl_coef | 0.1 | 高 |
| reward_scaling | 10 | 高 |
| gae_lambda | 0.95 | 中 |
| batch_size | 64 | 低 |

### DPO 关键超参数
| 参数 | 典型值 | 敏感度 |
|------|--------|--------|
| β | 0.1-0.5 | 中 |
| learning_rate | 1e-6 | 中 |
| batch_size | 64 | 低 |

**DPO 超参数少且更稳健**。

## 选择建议

### 选择 RLHF 的情况
- 需要在线学习（持续收集反馈）
- 有奖励模型基础设施
- 需要精细控制 KL 散度
- 复杂的多目标优化

### 选择 DPO 的情况
- 离线偏好数据充足
- 团队资源有限
- 需要快速实验迭代
- 不需要在线学习

## 组合使用

可以先用 RLHF 训练，再用 DPO 微调：
```python
# 1. RLHF 训练
model = train_rlhf(base_model, rlhf_data)

# 2. DPO 细调
model = train_dpo(model, dpo_data)
```

## 相关内容

- [[rlhf|RLHF 详解]]
- [[dpo|DPO 详解]]
- [[peft-methods-comparison|PEFT 方法对比]]
