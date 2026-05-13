---
title: 领域微调实践案例
created: 2026-04-23
updated: 2026-04-23
type: concept
tags: [concept, domain, finetuning, healthcare, legal, code]
sources: []
---

# 领域微调实践案例

## 一句话总结

领域微调的关键在于高质量领域数据 + 合理的微调策略，而非盲目扩大数据量。

## 医疗领域：MedPaLM / Meditron

### MedPaLM 2 (Google)

```yaml
基座模型: PaLM 2 (医学版)
微调方法:
  1. 持续预训练: 医学文献 + 临床文本
  2. 指令微调: 医学问答数据集
  3. 对齐: 医学专家偏好标注

数据规模:
  持续预训练: 数十亿医学 tokens
  指令微调: ~100K 医学问答对
  偏好对齐: ~10K 医学专家标注

效果:
  USMLE 考试: 86.5% (MedPaLM 2)
  医学问答准确率: 92.6%
  对比: 人类医生平均 ~87%
```

### Meditron (开源)

```yaml
基座模型: Llama 2-70B
微调方法: 持续预训练 + 医学指令微调

数据:
  PubMed 摘要: ~48M 条
  医学指南: ~1000 份
  临床问答: ~50K 条

训练配置:
  学习率: 2e-5
  批次大小: 128
  序列长度: 2048
  训练时长: ~150 GPU hours (A100)

关键技巧:
  - 使用医学实体识别过滤低质量数据
  - 分阶段训练: 预训练→指令微调→对齐
```

### 医疗微调代码示例

```python
# 医疗领域微调配置 (LLaMA Factory)
model_name_or_path: meta-llama/Llama-2-70b-hf
stage: sft
dataset: medical_qa
dataset_dir: data/medical

# 持续预训练阶段
finetuning_type: full
learning_rate: 2e-5
num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 16

# LoRA 阶段 (可选，降低显存)
# finetuning_type: lora
# lora_rank: 64
# lora_alpha: 128
```

## 代码领域：CodeLlama

### Meta CodeLlama

```yaml
基座模型: Llama 2
微调方法:
  1. 持续预训练: 500B tokens 代码数据
  2. 指令微调: 代码补全 + 代码解释任务

数据规模:
  主语言数据:
    - Python: 100B tokens
    - JavaScript: 80B tokens
    - Java: 60B tokens
    - C/C++: 60B tokens
  指令数据: ~50K 代码问答对

版本:
  - CodeLlama: 代码预训练版
  - CodeLlama-Python: Python 专项 (100B additional)
  - CodeLlama-Instruct: 指令微调版

效果 (HumanEval):
  CodeLlama-7B: pass@1 28.4%
  CodeLlama-13B: pass@1 35.1%
  CodeLlama-34B: pass@1 48.8%
  CodeLlama-70B: pass@1 53.7%
```

### Code 微调实践

```python
# 代码微调数据格式
{
  "instruction": "写一个Python函数，实现快速排序",
  "input": "",
  "output": """```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```"""
}

# 微调配置 (QLoRA)
model_name_or_path: codellama/CodeLlama-7b-hf
stage: sft
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
learning_rate: 3e-4
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
max_length: 4096
```

## 法律领域：ChatLaw

### 北大 ChatLaw

```yaml
基座模型: Baichuan-13B / Llama 2
微调方法:
  1. 法律知识注入: 法律条文 + 判例
  2. 法务任务微调: 合同审核、法条检索、案例分析

数据规模:
  法律条文: 全部现行法律 (约100万条)
  判例数据: ~200万份裁判文书摘要
  法务问答: ~50K 专业法务问答

训练配置:
  学习率: 1e-5
  训练轮数: 3 epochs
  批次大小: 32
  序列长度: 4096 (法律文件较长)

关键技巧:
  - 分领域微调: 民法/刑法/商法 分别训练
  - 检索增强: 先检索相关法条再生成回答
  - 温度和抽样调节: 降低幻觉
```

### 法律微调注意事项

```yaml
法律领域微调的挑战:
  1. 准确性要求高: 错误法条引用后果严重
  2. 知识更新快: 新法、新判例频繁
  3. 领域专业性: 需要法律专家标注

最佳实践:
  - 使用 RAG + 微调组合 (非纯微调)
  - 对法条索引单独训练检索模型
  - 保留法条引用的透明度
  - 定期更新底座知识
```

## 金融领域：FinGPT

### FinGPT (开源)

```yaml
基座模型: Llama 2 / Falcon
微调方法:
  1. 金融新闻预训练
  2. 金融报告理解
  3. 情感分析 (市场情绪)

数据:
  金融新闻: Reuters, Bloomberg 摘要
  财报: 美股10-K/10-Q报告
  金融情感: ~100K 标注样本
  新闻情感: ~50K 新闻情感标注

应用场景:
  - 财报解读
  - 市场情绪分析
  - 投资问答
  - 风险预警
```

### 金融微调配置

```yaml
# 金融领域 QLoRA
model_name_or_path: meta-llama/Llama-2-7b-hf
stage: sft
finetuning_type: lora
quantization_bit: 4
lora_rank: 32
lora_alpha: 64

dataset: financial_qa
learning_rate: 5e-5
num_train_epochs: 2
per_device_train_batch_size: 8
max_length: 2048

# 温度调低，减少幻觉
temperature: 0.3
```

## 通用领域微调配置模板

### 基础 SFT 配置

```yaml
# 基础微调配置 (适用于大多数领域)
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
stage: sft
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05

learning_rate: 5e-5
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
max_length: 2048
bf16: true

# 学习率调度
lr_scheduler_type: cosine
warmup_ratio: 0.1
```

### 数据量与训练轮数建议

| 数据量 | 推荐轮数 | 学习率 |
|--------|----------|--------|
| 1K-10K | 5-10 | 5e-5 |
| 10K-50K | 3-5 | 5e-5 |
| 50K-100K | 2-3 | 3e-5 |
| 100K+ | 1-2 | 2e-5 |

### 显存配置参考

| 模型规模 | LoRA | QLoRA (4bit) |
|----------|------|--------------|
| 7B | 16GB | 8GB |
| 13B | 24GB | 12GB |
| 30B | 48GB | 16GB |
| 70B | 80GB+ | 24GB |

## 领域微调关键决策

### 什么时候需要持续预训练

```yaml
需要持续预训练的场景:
  - 领域术语大量不同 (医疗、法律、金融)
  - 知识更新频繁 (医疗、金融)
  - 领域知识缺乏 (化学、物理学)
  - 特殊格式输出 (代码、数学证明)

只需指令微调的场景:
  - 通用对话增强
  - 指令遵循改进
  - 少量领域术语
  - 风格调整
```

### 微调前检查清单

```yaml
微调前确认:
  1. 数据量够吗? (建议 > 1K 高质量样本)
  2. 数据质量高吗? (宁愿少而精)
  3. 基座模型合适吗? (选对底座事半功倍)
  4. 评估基准定义了吗? (不定义如何改进)
  5. 显存够吗? (QLoRA 可以救急)
  6. 需要持续预训练吗? (看领域差距)
```

## 相关内容

- [[lora|LoRA]] — 低秩适配，领域微调标配
- [[qlora|QLoRA]] — 训练时量化，降低显存
- [[llama-factory|LLaMA Factory]] — 一站式微调框架
- [[unsloth|Unsloth]] — 极速微调
- [[rlhf|RLHF]] — 对齐微调方法
