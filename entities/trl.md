---
title: TRL: Transformer Reinforcement Learning
created: 2026-04-23
updated: 2026-04-23
type: entity
tags: [trl, framework, rlhf, dpo, ppo, grpo, kto, huggingface]
github: https://github.com/huggingface/trl
stars: 10000+
---

# TRL: Transformer Reinforcement Learning

## 概述

TRL 是 Hugging Face 开发的强化学习训练库，为大语言模型提供全栈的对齐训练工具。从 SFT 到 RLHF 到 DPO，TRL 提供了研究级别的实现和灵活的 API。

**核心价值**：
- Hugging Face 官方维护，生态集成最深
- 对齐方法覆盖最全面
- 纯 Python API，高度可定制
- 与 Transformers / PEFT / Datasets 无缝衔接

## 框架架构

```
┌──────────────────────────────────────────────────┐
│                   TRL API Layer                   │
├────────────┬────────────┬────────────┬────────────┤
│  SFTTrainer │ DPOTrainer │ PPOTrainer │ ...更多    │
│  监督微调    │ 偏好优化    │ 策略优化    │            │
├────────────┴────────────┴────────────┴────────────┤
│                 Training Core                      │
│  - 回滚机制 (Rollback)                              │
│  - 参考模型管理                                     │
│  - 奖励模型集成                                     │
│  - KL 散度约束                                     │
├───────────────────────────────────────────────────┤
│           HuggingFace 生态系统                      │
│  Transformers │ PEFT │ Accelerate │ Datasets       │
└───────────────────────────────────────────────────┘
```

## 支持的训练方法

| Trainer | 方法 | 论文/来源 |
|---------|------|-----------|
| SFTTrainer | 监督微调 | - |
| DPOTrainer | [[dpo\|DPO]] | Rafailov et al., 2023 |
| IPOTrainer | IPO | Azar et al., 2023 |
| KTOTrainer | [[kto\|KTO]] | Ethayarajh et al., 2024 |
| ORPOTrainer | ORPO | Hong et al., 2024 |
| CPOTrainer | CPO | Xu et al., 2024 |
| SimPOTrainer | SimPO | Meng et al., 2024 |
| PPOTrainer | [[rlhf\|RLHF]]/PPO | Ouyang et al., 2022 |
| GRPOTrainer | [[grpo\|GRPO]] | Shao et al., 2024 |
| RewardTrainer | 奖励模型训练 | - |
| AlignPropTrainer | AlignProp | - |

## 关键特性

### SFTTrainer — 监督微调
```python
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=2048,
    # 自动支持 LoRA
    peft_config=LoraConfig(r=16, lora_alpha=32),
)

trainer.train()
```

### DPOTrainer — 直接偏好优化
```python
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=preference_dataset,
    # beta 控制KL约束强度
    beta=0.1,
    loss_type="sigmoid",  # 或 "hinge", "ipo", "kto"
    peft_config=LoraConfig(r=16),
)

trainer.train()
```

### PPOTrainer — RLHF 训练
```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

config = PPOConfig(
    learning_rate=1e-5,
    batch_size=64,
    mini_batch_size=16,
    ppo_epochs=4,
    kl_coef=0.05,
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    reward_model=reward_model,
)

# 生成 → 评分 → 更新的循环
for batch in dataloader:
    response = ppo_trainer.generate(batch["query"])
    reward = reward_model(batch["query"], response)
    stats = ppo_trainer.step(batch["query"], response, reward)
```

### GRPOTrainer — 组相对策略优化
```python
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    num_generations=8,    # 每提示生成数
    per_device_train_batch_size=4,
    learning_rate=5e-7,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],  # 可多个奖励函数
    args=config,
    train_dataset=dataset,
)
trainer.train()
```

## 数据格式

### SFT 数据
```json
{"messages": [
  {"role": "user", "content": "问题"},
  {"role": "assistant", "content": "回答"}
]}
```

### 偏好数据 (DPO/KTO)
```json
{"prompt": "问题", "chosen": "好的回答", "rejected": "差的回答"}
```

### KTO 数据 (无需成对)
```json
{"prompt": "问题", "completion": "回答", "label": true}
```

## 与其他框架对比

| 特性 | TRL | [[axolotl\|Axolotl]] | [[llama-factory\|LLaMA Factory]] | [[unsloth\|Unsloth]] |
|------|-----|---------|---------|---------|
| API 风格 | Python 代码 | YAML | YAML + Web UI | Python 代码 |
| 对齐方法 | **最全** | 全 | 中 | 少 |
| 可定制性 | **最高** | 中 | 低 | 低 |
| DeepSpeed | 部分 | **最佳** | 支持 | 不支持 |
| 研究友好 | **最佳** | 中 | 低 | 低 |
| 上手难度 | 高 | 中 | **最低** | 低 |

## 适用场景

### 推荐 TRL 的情况
- RLHF/DPO/GRPO 研究实验
- 需要自定义训练循环
- 奖励模型训练与评估
- 需要组合多种对齐方法
- 学术论文复现

### 可选其他框架
- **[[llama-factory|LLaMA Factory]]**: 工程化生产部署
- **[[axolotl|Axolotl]]**: 大规模分布式训练
- **[[unsloth|Unsloth]]**: 快速原型实验

## 相关资源

- GitHub: https://github.com/huggingface/trl
- 文档: https://huggingface.co/docs/trl
- 教程: https://huggingface.co/learn/deep-rl-course

## 相关内容

- [[rlhf|RLHF]] — 人类反馈强化学习
- [[dpo|DPO]] — 直接偏好优化
- [[grpo|GRPO]] — 组相对策略优化
- [[kto|KTO]] — Kahneman-Tversky优化
- [[ppo|PPO]] — 近端策略优化
- [[axolotl|Axolotl]] — 另一微调框架
