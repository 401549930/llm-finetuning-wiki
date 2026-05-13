---
title: Llama 2 对齐训练
created: 2026-04-23
updated: 2026-04-23
type: entity
tags: [entity, model, meta, llama, rlhf, dpo, alignment]
sources: []
---

# Llama 2 对齐训练

## 一句话总结

Meta 的 Llama 2 是工业级 RLHF 对齐的标杆案例，展示了从 SFT 到多轮 RLHF 的完整管线，包括奖励模型迭代和安全性对齐。

## 训练总览

```
Llama 2 对齐管线:

阶段1: 监督微调 (SFT)
    └── 27,540 条高质量对话
        ↓
阶段2: 训练奖励模型 (RM)
    └── 1.4M 偏好对比数据
    └── 迭代训练多个版本
        ↓
阶段3: PPO 强化学习
    └── 5轮迭代 RLHF
    └── 每轮更新 RM + 策略
        ↓
阶段4: Ghost Attention (系统指令聚焦)
    └── 多轮对话系统指令对齐
        ↓
    Llama 2-Chat (最终版)
```

## 阶段1: SFT 详情

### 数据收集

```yaml
SFT 数据量: 27,540 条
来源:
  - 人工标注员编写高质量对话
  - 侧重: 有帮助性(helpfulness)对话
格式: |
  [INST] <<SYS>>
  {system_prompt}
  <</SYS>>

  {user_message} [/INST] {assistant_response}

数据质量要求:
  - 标注员需要确保回答有帮助、真实、安全
  - 排除低质量、重复、有毒回答
  - 多轮对话占比 > 60%
```

### 训练配置

```yaml
SFT 训练配置:
  模型: Llama 2 (7B/13B/34B/70B)
  训练轮数: 2 epochs (过拟合SFT对RLHF效果更好)
  学习率: 2e-5
  批次大小: 64
  序列长度: 4096
  权重衰减: 0.1
  Warmup: 100 steps
  余弦学习率调度
```

**关键决策：** SFT 2 epochs 效果优于 1 epoch — 过拟合 SFT 让模型学会格式，RLHF 再调整内容。

## 阶段2: 奖励模型

### 偏好数据

```yaml
偏好数据总量: 1,418,091 对
来源:
  - 人类标注员对比两个回答
  - 评判维度: 有帮助性 vs 安全性
  - 安全性 RM 和 有帮助性 RM 分别训练

数据分布:
  - 有帮助性: 1,040,750 对
  - 安全性: 377,341 对

标注流程:
  1. 给标注员展示同一提示的两个回答
  2. 标注员选择更好的那个 (或标记等价)
  3. 标注员解释选择理由
```

### RM 训练

```yaml
奖励模型训练:
  基座: Llama 2 (同规模) + 回归头
  损失: Bradley-Terry 偏好损失
  训练轮数: 1 epoch
  学习率: 5e-5 → 1e-6 (余弦衰减)
  批次大小: 512 pairs

  准确率 (验证集):
    有帮助性 RM: 73.2%
    安全性 RM: 67.5%
```

### 迭代 RM

```
第1轮 RLHF: RM v1 → PPO → 策略 v1
  ↓ 收集新偏好数据
第2轮 RLHF: RM v2 → PPO → 策略 v2
  ↓ 继续迭代
...
第5轮 RLHF: RM v5 → PPO → 策略 v5 (最终版)
```

每轮迭代都收集新的偏好数据，避免分布偏移。

## 阶段3: PPO 强化学习

### 训练配置

```yaml
PPO 训练配置:
  策略模型: Llama 2-Chat (SFT后)
  值函数: 与策略同规模
  奖励: 有帮助性RM + 安全性RM 加权

  超参数:
    PPO clip range: 0.2
    KL 惩罚系数: 0.01
    学习率: 5e-7
    批次大小: 512
    PPO 更新轮数: 1 per batch
    值函数裁剪: True

  奖励加权:
    helpful_score = helpfulness_rm_score
    safety_score = safety_rm_score
    total_reward = helpful_score + 0.1 * safety_score

  训练步数: ~1000 steps/轮
  总训练: 5轮迭代
```

### 安全性对齐

```python
# 安全性奖励逻辑 (简化)
def compute_reward(response, helpful_rm, safety_rm):
    h_score = helpful_rm(response)
    s_score = safety_rm(response)

    # 安全阈值：如果安全性分数低，大幅降低总奖励
    if s_score < threshold:
        return h_score * 0.1 + s_score * 10.0
    else:
        return h_score + 0.1 * s_score
```

## 阶段4: Ghost Attention

解决多轮对话中系统指令遗忘的问题：

```yaml
Ghost Attention (GAtt):
  问题: 模型在多轮对话后忘记系统指令
  方法:
    1. 在SFT数据中注入系统指令
    2. 训练时只在第一轮计算系统指令损失
    3. 后续轮次mask掉系统指令的loss
  效果: 系统指令在32+轮对话中仍然有效
```

## 效果指标

### 有帮助性 (MT-Bench)

| 模型 | 第一轮 | 第二轮 | 平均 |
|------|--------|--------|------|
| Llama 2-7B-Chat | 7.58 | 6.96 | 7.27 |
| Llama 2-13B-Chat | 7.91 | 7.30 | 7.60 |
| Llama 2-34B-Chat | 8.07 | 7.50 | 7.78 |
| Llama 2-70B-Chat | **8.37** | **7.81** | **8.09** |
| GPT-3.5-turbo | 8.18 | 7.61 | 7.89 |
| GPT-4 | 8.99 | 8.80 | 8.89 |

70B 版本超过 GPT-3.5-turbo。

### 安全性

| 模型 | 违规率 | 有帮助性保持 |
|------|--------|-------------|
| Llama 1-65B | 25.3% | - |
| Llama 2-7B-Chat | 8.1% | 7.27 |
| Llama 2-70B-Chat | **5.5%** | **8.09** |
| GPT-3.5-turbo | 5.2% | 7.89 |

## Llama 3 对齐改进

Llama 3 在 Llama 2 基础上做了关键改进：

```yaml
Llama 3 对齐变化:
  新增方法: DPO (直接偏好优化)
  SFT数据量: ~10M 条 (比Llama 2多300x)
  偏好数据: 多维度(事实性、指令遵循、语气)
  RLHF + DPO: 两者组合使用

  管线:
    阶段1: SFT (10M+ 条)
    阶段2: RLHF (PPO)
    阶段3: DPO (直接偏好优化)
    阶段4: 迭代对齐 (多轮)
```

## 可复现实践

### 用 TRL 复现 Llama 2 对齐

```python
# 阶段1: SFT
from trl import SFTTrainer
sft_trainer = SFTTrainer(
    model="meta-llama/Llama-2-7b-hf",
    train_dataset=sft_data,
    max_seq_length=4096,
    args=TrainingArguments(
        num_train_epochs=2,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
    ),
)

# 阶段2: 训练奖励模型
from trl import RewardTrainer
rm_trainer = RewardTrainer(
    model=rm_model,
    train_dataset=preference_data,
    args=TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=16,
    ),
)

# 阶段3: PPO
from trl import PPOTrainer
ppo_trainer = PPOTrainer(
    model=sft_model,
    reward_model=rm_model,
    train_dataset=prompts,
    args=PPOConfig(
        learning_rate=5e-7,
        batch_size=512,
        ppo_epochs=1,
        kl_coef=0.01,
    ),
)
```

## 经验总结

1. **SFT 可以过拟合** — 2 epochs 效果好于 1，RLHF 会修正过拟合
2. **迭代 RLHF 是关键** — 5轮迭代显著提升效果
3. **分离安全/有帮助性 RM** — 不同目标用不同奖励模型
4. **Ghost Attention 解决指令遗忘** — 简单但有效
5. **偏好数据量决定上限** — Llama 2 用 1.4M 对，Llama 3 用更多
6. **RLHF + DPO 组合更优** — Llama 3 的改进方向

## 相关内容

- [[rlhf|RLHF]] — 人类反馈强化学习
- [[ppo|PPO]] — 近端策略优化算法
- [[dpo|DPO]] — 直接偏好优化
- [[rl-methods-comparison|RL 对齐方法对比]]
- [[trl|TRL]] — HuggingFace训练库
