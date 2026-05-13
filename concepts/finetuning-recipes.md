---
title: 开源微调配方集
created: 2026-04-23
updated: 2026-04-23
type: concept
tags: [concept, recipe, finetuning, community, opensource]
sources: []
---

# 开源微调配方集

## 一句话总结

社区验证过的微调配方（Recipe），覆盖从数据准备到训练部署的完整流程，拿来即用。

## 配方1：中文对话模型 (Qwen2.5-7B)

```yaml
# 中文对话 QLoRA 微调
# 来源: 社区验证 | 显存: 16GB+ | 时间: ~30min (10K数据)

基座: Qwen/Qwen2.5-7B-Instruct
框架: LLaMA Factory / Unsloth
方法: QLoRA

数据准备:
  格式: ShareGPT / Alpaca
  数据集: 
    - bellegroup: 50K 中文对话
    - firefly: 160K 中文多任务
    - 自有数据: 5K+ 领域问答
  清洗:
    - 去除 < 10 字的回复
    - 去除重复问题
    - 保留高质量长回复

训练配置:
  quantization_bit: 4
  lora_rank: 64
  lora_alpha: 128
  lora_target: all
  learning_rate: 5e-5
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  max_length: 2048
  warmup_ratio: 0.1
  lr_scheduler_type: cosine

评估:
  - 人工对比测试 (50条)
  - C-Eval 中文评测
  - CMMLU 中文多任务

部署:
  - 合并 LoRA → GGUF Q5_K_M
  - Ollama 部署本地服务
  - vLLM 部署 API 服务
```

## 配方2：代码助手 (CodeLlama-13B)

```yaml
# 代码助手 QLoRA 微调
# 来源: 社区验证 | 显存: 24GB+ | 时间: ~1h (20K数据)

基座: codellama/CodeLlama-13b-Instruct-hf
框架: Axolotl / Unsloth
方法: QLoRA

数据准备:
  格式: 指令格式
  数据集:
    - Code Alpaca: 20K 代码指令
    - 自有代码库: 提取函数 + 文档对
    - LeetCode 解题: 5K 带注释
  增强:
    - 函数签名 → 实现
    - Bug 描述 → 修复
    - 注释 → 代码

训练配置:
  quantization_bit: 4
  lora_rank: 128              # 代码需要更高秩
  lora_alpha: 256
  lora_target: all
  learning_rate: 3e-4         # 代码微调可用更高学习率
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_length: 4096            # 代码需要更长上下文

关键技巧:
  - EOS token 设为代码结束符
  - 保留缩进格式
  - 多语言混合训练
  - 加入单元测试数据
```

## 配方3：DPO 对齐 (Llama-3-8B)

```yaml
# DPO 对齐微调
# 来源: 社区验证 | 显存: 24GB+ | 时间: ~2h

基座: meta-llama/Meta-Llama-3-8B-Instruct
框架: TRL / Axolotl
方法: QLoRA + DPO

第一步: SFT (可选，如已有SFT模型可跳过)
  stage: sft
  quantization_bit: 4
  lora_rank: 64
  lora_alpha: 128
  learning_rate: 5e-5
  num_train_epochs: 2

第二步: DPO 对齐
  stage: dpo
  quantization_bit: 4
  lora_rank: 64
  lora_alpha: 128
  learning_rate: 5e-6        # DPO 用更低学习率
  num_train_epochs: 1        # 1轮通常够用
  dpo_beta: 0.1              # KL 惩罚系数
  dpo_loss_type: sigmoid     # 标准 DPO 损失

数据格式:
  {
    "prompt": "解释量子计算的基本原理",
    "chosen": "量子计算利用量子叠加和纠缠...",
    "rejected": "量子计算就是很快的计算机..."
  }

数据来源:
  - Self-Instruct: 用模型生成多回答 + 人工排序
  - UltraFeedback: 60K 偏好对 (开源)
  - Argilla: 社区标注偏好数据

效果验证:
  - MT-Bench 分数对比
  - 人类偏好 A/B 测试
  - 安全性测试 (red team)
```

## 配方4：GRPO 推理模型 (Qwen2.5-7B)

```yaml
# GRPO 推理能力训练
# 参考: DeepSeek-R1 | 显存: 24GB+ | 时间: ~4h

基座: Qwen/Qwen2.5-7B-Instruct
框架: TRL (GRPOTrainer) / LLaMA Factory
方法: GRPO (无标注数据)

第一步: 冷启动 SFT
  stage: sft
  data: ~500 条长 CoT 推理示例
  lora_rank: 64
  learning_rate: 5e-5
  num_train_epochs: 3

第二步: GRPO 强化
  stage: grpo
  grpo_num_generations: 16    # 每提示生成K个回答
  learning_rate: 5e-7
  num_train_epochs: 1
  max_completion_length: 2048  # 允许长推理链

奖励函数 (规则):
  数学: 提取答案 + 比对标准答案
  代码: 执行 + 比对测试用例
  格式: 检查 CoT 格式完整性

数据:
  数学: GSM8K / MATH 提示集
  代码: HumanEval / MBPP 提示集
  逻辑: 自编逻辑推理题

注意事项:
  - 需要足够大的 K (至少8)
  - 奖励函数必须可靠
  - 需要较长训练时间
  - 建议从 7B 开始实验
```

## 配方5：多模态微调 (LLaVA)

```yaml
# 视觉语言模型微调
# 显存: 24GB+ | 时间: ~2h

基座: llava-hf/llava-1.5-7b-hf
框架: LLaMA Factory
方法: QLoRA

数据准备:
  格式: 多模态对话
  {
    "messages": [
      {"role": "user", "content": [
        {"type": "image", "image": "path/to/image.jpg"},
        {"type": "text", "text": "描述这张图片"}
      ]},
      {"role": "assistant", "content": [
        {"type": "text", "text": "这张图片显示..."}
      ]}
    ]
  }

训练配置:
  stage: sft
  quantization_bit: 4
  lora_rank: 64
  lora_alpha: 128
  learning_rate: 3e-5
  num_train_epochs: 2
  max_length: 2048
  image_resolution: 336

应用场景:
  - OCR + 文档理解
  - 医学影像解读
  - 产品图片描述
```

## 配方6：LoRA 合并 (模型融合)

```yaml
# 多个 LoRA 适配器合并
# 无需训练，仅推理

场景: 同时需要多种能力的模型
  - LoRA-A: 数学能力
  - LoRA-B: 代码能力
  - LoRA-C: 中文能力

合并方法:
  1. TIES 合并 (推荐)
  2. DARE 合并
  3. 线性加权合并

# LLaMA Factory 合并
merge_method: ties
adapter_name_or_path:
  - ./lora-math
  - ./lora-code
  - ./lora-chinese
merge_weight: [0.4, 0.3, 0.3]  # 权重可调

# 或使用 mergekit
models:
  - model: base-model
  - model: base-model + lora-math
    parameters:
      density: 0.5
      weight: 0.4
  - model: base-model + lora-code
    parameters:
      density: 0.5
      weight: 0.3
merge_method: ties
```

## 数据准备速查

### 数据格式转换

```python
# 各种格式互转

# Alpaca → ShareGPT
def alpaca_to_sharegpt(item):
    return {
        "conversations": [
            {"from": "human", "value": item["instruction"]},
            {"from": "gpt", "value": item["output"]},
        ]
    }

# ShareGPT → LLaMA Factory
def sharegpt_to_llamafactory(item):
    messages = []
    for conv in item["conversations"]:
        role = "user" if conv["from"] == "human" else "assistant"
        messages.append({"role": role, "content": conv["value"]})
    return {"messages": messages}

# 偏好数据格式 (DPO)
def to_dpo_format(prompt, chosen, rejected):
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }
```

### 数据质量检查

```python
# 数据质量检查脚本
import json
from collections import Counter

def check_dataset(file_path):
    with open(file_path) as f:
        data = [json.loads(line) for line in f]

    print(f"总样本数: {len(data)}")
    
    # 检查长度分布
    lengths = [len(item.get("output", "")) for item in data]
    print(f"输出长度: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")
    
    # 检查重复
    prompts = [item.get("instruction", "") for item in data]
    print(f"重复提示: {len(prompts) - len(set(prompts))}")
    
    # 检查空值
    empty = sum(1 for item in data if not item.get("output", "").strip())
    print(f"空输出: {empty}")
```

## 训练监控

### 关键指标

```yaml
训练时关注:
  loss:
    - 持续下降: 正常
    - 不降反升: 学习率过大或数据问题
    - 震荡剧烈: batch size 太小

  eval_loss:
    - 与 train_loss 差距大: 过拟合
    - 两者同步下降: 正常

  learning_rate:
    - 确认余弦调度正常
    - warmup 阶段逐渐上升

  GPU 利用率:
    - < 70%: 可能 batch size 太小
    - > 95%: 可能接近 OOM
```

### Weights & Biases 集成

```yaml
# 启用 W&B 记录
report_to: wandb
run_name: qwen-7b-qlora-v1
wandb_project: llm-finetuning

# LLaMA Factory
use_wandb: true
wandb_run_name: qwen-7b-qlora
```

## 常见踩坑

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 训练 loss 不降 | 学习率太小 | 提高到 5e-5 或 1e-4 |
| 训练 loss 爆炸 | 学习率太大 | 降低到 1e-5 |
| eval_loss 远高于 train_loss | 过拟合 | 减少轮数/加 dropout |
| 模型输出重复 | 温度太低或数据问题 | 检查数据/调高温度 |
| 中文能力下降 | 英文数据过多 | 平衡中英文数据比例 |
| 格式崩坏 | LoRA rank 太小 | 增大到 64 或 128 |
| 显存 OOM | batch 太大 | 减小 batch + 增大 accum |
| 训练很慢 | 未开 Flash Attention | 启用 flash_attention_2 |

## 相关内容

- [[qlora-practice|QLoRA 实战配置]] — QLoRA 详细配置指南
- [[domain-finetuning|领域微调实践案例]] — 医疗/法律/代码/金融
- [[llama-factory|LLaMA Factory]] — 一站式微调框架
- [[unsloth|Unsloth]] — 极速微调
- [[trl|TRL]] — 灵活训练库
