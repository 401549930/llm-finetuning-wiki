---
title: RLHF: Reinforcement Learning from Human Feedback
created: 2026-04-23
updated: 2026-04-23
type: method
tags: [rlhf, alignment, reinforcement-learning]
sources: [raw/papers/arxiv-2203.02155.md]
arxiv_id: "2203.02155"
---

# RLHF: Reinforcement Learning from Human Feedback

## 一句话总结
通过人类反馈训练奖励模型，再用强化学习优化语言模型，使其对齐人类意图。

## 核心原理

RLHF 分为三个阶段：

### 阶段1: 监督微调 (SFT)
- 收集人类撰写的示范回答
- 用监督学习微调预训练模型
- 得到 SFT 模型

### 阶段2: 奖励模型训练 (RM)
- 让模型生成多个回答
- 人类对回答进行排序
- 训练奖励模型学习人类偏好

### 阶段3: 强化学习优化 (PPO)
- 用奖励模型评分
- 用 PPO 算法优化策略
- 加入 KL 散度约束防止偏离

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 预训练模型   │ --> │ SFT 微调    │ --> │ PPO 优化    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                    │
                           v                    v
                    ┌─────────────┐     ┌─────────────┐
                    │ 人类示范数据 │     │ 奖励模型    │
                    └─────────────┘     └─────────────┘
                                              ^
                                              │
                                       ┌─────────────┐
                                       │ 人类排序数据 │
                                       └─────────────┘
```

## InstructGPT 实现

论文使用 GPT-3 作为基座：

| 组件 | 规模 | 说明 |
|------|------|------|
| SFT 模型 | 175B | 基于预训练 GPT-3 |
| RM 模型 | 6B | 移除最后一层的 GPT-3 |
| PPO | 1.3B/6B/175B | 多规模实验 |

关键发现：**1.3B InstructGPT 的输出优于 175B GPT-3**

## 优缺点

| 优点            | 缺点          |
| ------------- | ----------- |
| 显著提升对齐质量      | 需要大量人类标注    |
| 减少有害输出        | 训练流程复杂      |
| 可迁移到多种任务      | PPO 不稳定，难调参 |
| 奠定 ChatGPT 基础 | 奖励模型可能被"欺骗" |

## 训练细节

### 奖励模型
- 输入：(prompt, response) 对
- 输出：标量奖励值
- 损失函数：Bradley-Terry 模型

$$\mathcal{L} = -\mathbb{E}[\log \sigma(r(x, y_w) - r(x, y_l))]$$

其中 $y_w$ 是更优回答，$y_l$ 是较差回答。

### PPO 目标函数

$$\mathcal{L}_{PPO} = \mathbb{E}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

加入 KL 惩罚：

$$\mathcal{L}_{total} = \mathcal{L}_{PPO} - \beta \cdot \mathbb{E}[D_{KL}(\pi_\theta \| \pi_{ref})]$$

## 相关方法

- [[dpo|DPO]]：直接优化偏好，无需奖励模型
- [[ppo|PPO]]：RLHF 使用的强化学习算法
- [[grpo|GRPO]]：改进的 RL 方法

## 主要论文

- [[raw/papers/arxiv-2203.02155|Training language models to follow instructions with human feedback]] (Ouyang et al., 2022)

## 参考文献

- OpenAI Blog: https://openai.com/research/instruction-following
- Deep RL from Human Preferences (Christiano et al., 2017)
- Learning to Summarize from Human Feedback (Stiennon et al., 2020)
