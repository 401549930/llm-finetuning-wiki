---
title: PPO: Proximal Policy Optimization
created: 2026-04-23
updated: 2026-04-23
type: method
tags: [ppo, reinforcement-learning, rlhf, policy-gradient]
sources: []
paper: "Proximal Policy Optimization Algorithms"
arxiv_id: "1707.06347"
---

# PPO: Proximal Policy Optimization

## 一句话总结
通过限制策略更新幅度实现稳定的强化学习训练，是 [[rlhf|RLHF]] 中最常用的策略优化算法。

## 核心原理

PPO 解决了策略梯度方法的核心痛点：更新步长太大导致策略崩塌，步长太小学习太慢。

### 关键创新：Clip 目标函数

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是概率比
- $\hat{A}_t$ 是优势估计
- $\epsilon$ 是裁剪范围 (通常 0.1-0.2)

### 为什么需要 Clip？

```
奖励
  │
  │     新策略（更新太大）
  │    ╱╲
  │   ╱  ╲  可能进入糟糕区域
  │  ╱    ╲
  │ ╱ 旧策略╲
  │╱_________╲________ 策略空间
  │
  │  PPO 通过 clip 限制更新幅度
  │  确保新策略在旧策略附近
```

## PPO 算法流程

```
初始化策略 π_θ

for iteration:
    1. 用当前策略收集轨迹数据
    2. 计算优势估计 Â (用 GAE)
    3. 多轮更新策略:
       for epoch in range(K):  # K=3-10
           计算概率比 r(θ)
           计算 clip 目标 L_CLIP
           反向传播更新 θ
    4. 可选: 更新价值函数
```

### 优势估计 (GAE)

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

## 在 RLHF 中的应用

### InstructGPT / ChatGPT 训练流程

```
┌─────────────────────────────────────────────┐
│  阶段 1: SFT (监督微调)                       │
│  用人类示范数据微调预训练模型                  │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  阶段 2: 训练奖励模型                         │
│  用人类排序数据训练奖励预测器                  │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  阶段 3: PPO 优化 ← 本算法                    │
│  用奖励模型评分 + KL 约束优化策略              │
└─────────────────────────────────────────────┘
```

### RLHF 目标函数

$$\mathcal{L}_{RLHF} = \mathcal{L}_{PPO} - \beta \cdot \mathbb{E}[D_{KL}(\pi_\theta \| \pi_{ref})]$$

KL 约束防止策略偏离太远，保持语言能力。

## 实现示例

### TRL PPOTrainer
```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

config = PPOConfig(
    learning_rate=1e-5,
    batch_size=64,
    mini_batch_size=16,
    ppo_epochs=4,           # K 轮更新
    cliprange=0.2,          # ε
    kl_coef=0.05,           # β
    gamma=1.0,              # 折扣因子
    lam=0.95,               # GAE λ
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,    # KL 参考模型
    tokenizer=tokenizer,
    reward_model=reward_model,
)

# 训练循环
for batch in dataloader:
    query_tensors = batch["input_ids"]

    # 生成回答
    response_tensors = ppo_trainer.generate(query_tensors)

    # 计算奖励
    rewards = reward_model(query_tensors, response_tensors)

    # PPO 更新
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

## 超参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `cliprange` (ε) | 0.1-0.2 | 裁剪范围，越大允许越大更新 |
| `kl_coef` (β) | 0.02-0.1 | KL 约束强度 |
| `ppo_epochs` | 3-10 | 每批数据更新轮数 |
| `learning_rate` | 1e-6 to 1e-5 | 学习率，RLHF 需要较小 |
| `gamma` | 1.0 | 折扣因子，语言任务通常为1 |
| `lam` (GAE) | 0.95 | 方差-偏差权衡 |

## 与其他 RL 方法对比

| 方法 | 稳定性 | 样本效率 | 实现复杂度 | RLHF 适用 |
|------|--------|----------|------------|-----------|
| PPO | **高** | 中 | 中 | ✅ 主流 |
| A3C | 低 | 高 | 高 | ❌ |
| TRPO | 高 | 低 | **最高** | 部分 |
| REINFORCE | 低 | 低 | 低 | ❌ |

## 优缺点

| 优点 | 缺点 |
|------|------|
| 训练稳定 | 需要 KL 约束 |
| 样本效率较好 | 奖励模型质量关键 |
| 广泛验证 | 超参数敏感 |
| 实现相对简单 | 可能被奖励欺骗 |

## 相关论文

- Proximal Policy Optimization Algorithms (Schulman et al., 2017)
- Trust Region Policy Optimization (Schulman et al., 2015)
- Training language models to follow instructions (Ouyang et al., 2022)

## 相关内容

- [[rlhf|RLHF]] — PPO 在 RLHF 中的应用
- [[dpo|DPO]] — 无需 PPO 的替代方案
- [[grpo|GRPO]] — 简化版 RL 方法
- [[trl|TRL]] — PPO 训练实现
