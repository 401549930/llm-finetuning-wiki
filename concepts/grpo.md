---
title: GRPO: Group Relative Policy Optimization
created: 2026-04-23
updated: 2026-04-23
type: method
tags: [grpo, rl, alignment, rlhf, ppo]
sources: []
paper: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
---

# GRPO: Group Relative Policy Optimization

## 一句话总结
通过组内相对比较简化奖励模型，让 LLM 自主学习复杂推理能力，无需 SFT 预训练和人工标注。

## 核心原理

GRPO 是 DeepSeek-R1 论文中提出的强化学习方法，核心创新是用**组内相对比较**替代独立奖励模型。

### 与 PPO/RLHF 的区别

| 方面 | [[rlhf\|RLHF]]/PPO | GRPO |
|------|-------------------|------|
| 奖励来源 | 独立训练的奖励模型 | 组内相对排序 |
| SFT 预训练 | 必需 | **不需要** |
| KL 约束 | 需要 | **不需要** |
| 计算开销 | 高 (双模型) | 低 (单模型) |

### 算法流程

```
对于每个提示 x:
1. 从当前策略 π_θ 采样 K 个回答 {y_1, ..., y_K}
2. 用规则奖励函数 R(x, y) 计算每个回答的奖励
3. 组内归一化: A_i = (r_i - mean(r)) / std(r)
4. 用相对优势优化策略
```

### 目标函数

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[\frac{1}{K}\sum_{i=1}^{K} A_i \cdot \log \pi_\theta(y_i|x)\right]$$

其中优势值 $A_i$ 基于组内相对排序计算：
$$A_i = \frac{r_i - \text{mean}(\{r_1,...,r_K\})}{\text{std}(\{r_1,...,r_K\})}$$

## 关键创新

### 1. 消除奖励模型
- 不需要训练独立的奖励模型
- 使用规则奖励 (如正确性、格式)
- 组内相对比较作为监督信号

### 2. 无需 SFT 预训练
- DeepSeek-R1-Zero 直接从基座模型开始 RL
- 仅用规则奖励引导学习
- 模型自主发展出推理能力

### 3. 无 KL 约束
- 不需要约束策略偏离
- 更自由地探索解空间

## 实现示例

```python
import torch
from trl import GRPOTrainer, GRPOConfig

# GRPO 配置
config = GRPOConfig(
    num_generations=8,      # 每提示生成数 K
    beta=0.04,              # KL 系数 (可选)
    per_device_train_batch_size=4,
    learning_rate=1e-6,
)

# 定义规则奖励函数
def reward_fn(prompts, responses):
    rewards = []
    for p, r in zip(prompts, responses):
        # 示例: 检查格式正确性
        if "因此" in r and len(r) > 100:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return torch.tensor(rewards)

# 训练
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=config,
    train_dataset=dataset,
)
trainer.train()
```

## DeepSeek-R1 应用

### 训练流程
```
基座模型 (DeepSeek-V3-Base)
    │
    ├── GRPO 强化学习 (无需 SFT)
    │   └── 数学/代码任务 + 规则奖励
    │
    └── DeepSeek-R1-Zero
        │
        └── 长链推理能力涌现
```

### 涌现能力
- 自动学会"先思考再回答"
- 发展出自我验证能力
- 长链推理 (思维链)
- 多步骤问题分解

## 与其他 RL 方法对比

| 方法 | 需要奖励模型 | 需要 SFT | 需要 KL 约束 | 训练稳定性 |
|------|-------------|----------|--------------|------------|
| [[rlhf\|RLHF]]/PPO | ✅ | ✅ | ✅ | 低 |
| [[dpo\|DPO]] | ❌ (用偏好数据) | ✅ | ❌ | 高 |
| [[kto\|KTO]] | ❌ (用单点标注) | ✅ | ❌ | 高 |
| GRPO | ❌ (用规则奖励) | ❌ | ❌ | **中高** |

## 优缺点

| 优点 | 缺点 |
|------|------|
| 无需标注数据 | 需要设计规则奖励 |
| 不需要 SFT 预训练 | 训练时间可能较长 |
| 计算开销低 | 依赖规则质量 |
| 可涌现推理能力 | 可能产生格式漂移 |

## 适用场景

- 数学/代码推理任务 (有客观答案)
- 不想投入标注成本时
- 希望模型涌现推理能力
- 从基座模型直接训练

## 相关论文

- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2025)
- DeepSeekMath: Prospective Granting (2024)

## 相关内容

- [[rlhf|RLHF]] — 人类反馈强化学习
- [[dpo|DPO]] — 直接偏好优化
- [[kto|KTO]] — Kahneman-Tversky优化
- [[ppo|PPO]] — 近端策略优化
- [[trl|TRL]] — HuggingFace训练库，支持GRPO
