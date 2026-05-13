---
title: DeepSeek-R1 微调实践
created: 2026-04-23
updated: 2026-04-23
type: entity
tags: [entity, model, deepseek, grpo, rlhf, reasoning]
sources: [raw/papers/arxiv-2501.12948]
---

# DeepSeek-R1 微调实践

## 一句话总结

DeepSeek-R1 通过纯 RL 训练（GRPO）涌现出推理能力，证明无需 SFT 预热也能激活 CoT 推理，是 RL 对齐方法的里程碑案例。

## 两条训练路线

### DeepSeek-R1-Zero：纯 RL 路线

```
基座模型 (DeepSeek-V3-Base)
    │
    └──→ GRPO 强化学习（无SFT预热）
         ├── 奖励函数：规则评分（准确性 + 格式）
         ├── 无需任何标注数据
         └── 涌现推理行为：反思、验证、多路径探索
```

**关键发现：** 纯 RL 训练自发涌现了：
- 主动反思 (self-verification)
- 多路径探索
- 答案验证行为
- Chain-of-Thought 推理链

**问题：** 输出格式不稳定，语言混用（中英混杂）

### DeepSeek-R1：多阶段路线

```
阶段1: 冷启动 SFT
    └── 少量高质量长 CoT 示例 (~8000条)
        ↓
阶段2: GRPO 推理强化
    └── 规则奖励（数学、代码、逻辑推理）
        ↓
阶段3: 拒绝采样 SFT
    └── 用阶段2模型生成数据 + 通用SFT数据
    └── 重新训练基座模型
        ↓
阶段4: 全场景 DPO
    └── 推理数据 + 非推理数据对齐
        ↓
    DeepSeek-R1 (最终版)
```

## 训练配置详解

### GRPO 配置

```yaml
# DeepSeek-R1 GRPO 训练配置
算法: GRPO (Group Relative Policy Optimization)
每组采样数 (K): 64
奖励类型: 规则奖励（非神经网络）

数学奖励:
  - 正确答案: +1.0
  - 格式正确但答案错: 0.0
  - 格式错误: -0.5

代码奖励:
  - 通过测试用例数 / 总测试用例数

KL 散度惩罚: 0.001 (非常小，允许策略偏移)
```

### 冷启动数据

```yaml
数据量: ~8000 条高质量长CoT推理示例
来源:
  - few-shot prompting 生成
  - 人工筛选和格式修正
格式: |
  <think>
  [详细推理过程，可能包含反思、验证、回溯]
  </think>
  [最终答案]
```

### 拒绝采样 SFT

```yaml
阶段3 SFT 数据:
  推理数据: ~600K (从阶段2模型拒绝采样生成)
  通用数据: ~200K (写作、问答、翻译等)
  总计: ~800K 条

拒绝采样策略:
  - 生成多个回答
  - 用规则/奖励模型筛选正确回答
  - 仅保留高质量样本用于SFT
```

### DPO 对齐

```yaml
阶段4 DPO 配置:
  推理偏好对: 从GRPO模型采样 + 规则评分排序
  通用偏好对: 人工标注 + 模型辅助标注
  学习率: 5e-7
  训练轮数: 1-2 epochs
```

## 效果对比

### 数学推理

| 模型 | MATH-500 | AIME 2024 | GPQA Diamond |
|------|----------|-----------|--------------|
| DeepSeek-V3 (基座) | 63.6% | 23.3% | 51.1% |
| DeepSeek-R1-Zero | 73.0% | 36.0% | 62.0% |
| DeepSeek-R1 | **82.6%** | **50.0%** | **68.4%** |
| OpenAI o1 | 96.4% | 74.3% | 76.4% |

### 代码推理

| 模型 | LiveCodeBench | Codeforces 评分 |
|------|---------------|----------------|
| DeepSeek-V3 | 28.9% | ~1200 |
| DeepSeek-R1 | **42.3%** | **~1800** |

## 蒸馏实验

DeepSeek-R1 还验证了蒸馏的有效性：

```
DeepSeek-R1 (671B MoE)
    │
    └──→ 蒸馏到小模型
         ├── Qwen-1.5B → AIME 28.9%
         ├── Qwen-7B → AIME 33.3%
         ├── Qwen-14B → AIME 40.0%
         ├── Qwen-32B → AIME 46.7%
         └── Llama-70B → AIME 46.7%
```

**关键发现：** 蒸馏比直接在小模型上跑RL更有效。

## 可复现实践

### 用 GRPO 训练推理模型

```python
# 使用 TRL 的 GRPO Trainer
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    output_dir="./r1-grpo",
    num_iterations=3,           # GRPO 迭代次数
    per_device_train_batch_size=4,
    num_generations=16,         # K=16 (显存有限时减少)
    learning_rate=5e-7,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    max_completion_length=2048, # 允许长推理链
)

def reward_fn(completions, **kwargs):
    """规则奖励函数"""
    rewards = []
    for comp in completions:
        # 提取 <think> 和答案
        if has_valid_format(comp):
            answer = extract_answer(comp)
            if answer == kwargs.get("ground_truth"):
                rewards.append(1.0)
            else:
                rewards.append(-0.1)
        else:
            rewards.append(-0.5)
    return rewards

trainer = GRPOTrainer(
    model=model,
    config=config,
    reward_funcs=reward_fn,
    train_dataset=dataset,
)
trainer.train()
```

### 用 LLaMA Factory 复现

```yaml
# LLaMA Factory GRPO 配置
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
stage: grpo
reward_model_type: rule        # 规则奖励
grpo_num_generations: 16       # 每提示生成数
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5e-7
num_train_epochs: 1
bf16: true
max_length: 4096
```

## 经验总结

1. **GRPO 比 PPO 更实用** — 无需训练 critic model，节省一半计算
2. **冷启动 SFT 很关键** — 纯 RL 训练格式不稳定，少量 SFT 数据即可解决
3. **规则奖励够用** — 推理任务不需要神经网络奖励模型
4. **拒绝采样 > 直接RL** — 用 RL 模型生成数据再做 SFT 更稳定
5. **蒸馏优于小模型RL** — 把大模型推理能力蒸馏给小模型更高效
6. **长输出很重要** — 允许模型生成超长推理链（max_completion_length >= 2048）

## 相关内容

- [[grpo|GRPO]] — 组相对策略优化算法
- [[dpo|DPO]] — 直接偏好优化
- [[rlhf|RLHF]] — 人类反馈强化学习
- [[rl-methods-comparison|RL 对齐方法对比]]
- [[llama-factory|LLaMA Factory]] — 支持GRPO的框架
