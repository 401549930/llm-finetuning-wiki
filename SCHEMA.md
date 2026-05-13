# Wiki Schema — LLM Fine-tuning

## Domain
此知识库专注于大语言模型微调方法、论文和框架：
- 微调技术：LoRA, QLoRA, PEFT, Adapter, Prefix Tuning 等
- RL方法：RLHF, DPO, PPO, GRPO, KTO 等
- 训练框架：Axolotl, TRL, Unsloth, LLaMA-Factory, Firefinch 等
- 量化方法：GGUF, GPTQ, AWQ, bitsandbytes 等
- 基准评测：MMLU, HumanEval, MT-Bench 等
- 相关论文和研究进展

## Conventions
- 文件名：小写、连字符、无空格（如 `lora-finetuning.md`）
- 每个页面必须有 YAML frontmatter
- 使用 `[[wikilinks]]` 链接相关页面（每页至少2个出链）
- 更新页面时必须更新 `updated` 日期
- 新页面必须添加到 `index.md` 对应分区
- 所有操作必须记录到 `log.md`

## Frontmatter
```yaml
---
title: 页面标题
created: YYYY-MM-DD
updated: YYYY-MM-DD
type: entity | concept | comparison | query | paper | method
tags: [从下方分类选取]
sources: [raw/articles/source-name.md]
---
```

## Tag Taxonomy

### 微调方法
- `lora` - Low-Rank Adaptation 相关
- `qlora` - 量化 LoRA
- `peft` - Parameter-Efficient Fine-Tuning 通用
- `adapter` - Adapter 系列方法
- `prefix-tuning` - 前缀微调
- `prompt-tuning` - 提示微调
- `full-finetuning` - 全参数微调

### RL 方法
- `rlhf` - 人类反馈强化学习
- `dpo` - Direct Preference Optimization
- `ppo` - Proximal Policy Optimization
- `grpo` - Group Relative Policy Optimization
- `kto` - Kahneman-Tversky Optimization
- `spo` - Simple Preference Optimization

### 训练框架
- `axolotl` - Axolotl 训练框架
- `trl` - Hugging Face TRL
- `unsloth` - Unsloth 高效训练
- `llama-factory` - LLaMA-Factory
- `firefly` - Firefly（流萤）
- `deepspeed` - DeepSpeed 分布式训练
- `fsdp` - Fully Sharded Data Parallel

### 量化
- `gguf` - GGUF 格式
- `gptq` - GPTQ 量化
- `awq` - AWQ 量化
- `bnb` - bitsandbytes

### 数据与评测
- `dataset` - 训练数据集
- `benchmark` - 评测基准
- `evaluation` - 模型评测方法

### 论文与来源
- `arxiv` - arXiv 论文
- `neurips` - NeurIPS 会议
- `iclr` - ICLR 会议
- `acl` - ACL 会议
- `emnlp` - EMNLP 会议

### 内容类型
- `paper` - 论文笔记
- `method` - 方法详解
- `comparison` - 方法对比
- `tutorial` - 教程指南
- `entity` - 机构/人物/产品

## Page Thresholds
- **创建页面**：实体/概念出现于2+源 或 在单一源中为核心内容
- **添加到现有页面**：源材料提及已有实体/概念
- **不创建页面**：仅提及的名词、次要细节、领域外内容
- **拆分页面**：超过200行时分拆为子主题并交叉链接
- **归档页面**：内容完全过时时移动至 `_archive/`

## Entity Pages (实体页)
- 人物：研究人员、开源贡献者
- 机构：实验室、公司、开源社区
- 框架/产品：训练框架名称
- 模型：重要基座模型或微调模型

## Concept Pages (概念页)
- 理论基础与定义
- 当前研究进展
- 开放问题与争议
- 相关概念链接

## Paper Pages (论文页)
- 核心贡献总结
- 方法细节
- 实验结果
- 后续影响与引用

## Comparison Pages (对比页)
- 对比维度表格
- 各方法优劣
- 适用场景
- 综合结论

## Update Policy
当新信息与现有内容冲突时：
1. 检查日期 — 新来源通常优先
2. 真正矛盾时，标注两种观点（含日期和来源）
3. 在 frontmatter 标记：`contradictions: [page-name]`
4. 在 lint 报告中标记供用户审核
