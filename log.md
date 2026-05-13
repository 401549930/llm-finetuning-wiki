# Wiki Log

> 所有知识库操作的时间线记录。仅追加。
> 格式: `## [YYYY-MM-DD] action | subject`
> 操作类型: ingest, update, query, lint, create, archive, delete
> 文件超过500条时，轮换为 log-YYYY.md

## [2026-04-23] create | 大模型微调知识库初始化
- 知识库路径: /mnt/c/Users/76750/Documents/projects/obsidian_projects/llm-finetuning-wiki
- 领域: 大模型微调方法与论文
- 目录结构:
  - raw/ (articles, papers, transcripts, assets)
  - entities/
  - concepts/
  - comparisons/
  - queries/
  - _templates/
  - scripts/
- 核心文件:
  - SCHEMA.md (领域规范、标签分类)
  - index.md (内容目录)
  - log.md (操作日志)
- 专注内容:
  - PEFT 方法 (LoRA, QLoRA, Adapter)
  - RL 方法 (RLHF, DPO, GRPO)
  - 训练框架 (Axolotl, TRL, Unsloth)
  - 量化技术 (GGUF, GPTQ, AWQ)
  - 微调论文与研究进展

## [2026-04-23] ingest | LLaMA Factory 论文与教程文章
- 来源: 
  - arXiv 论文 2403.13372 (ACL 2024)
  - VoltAgent 博客教程
- 导入文件:
  - raw/papers/arxiv-2403.13372.md — 论文原文摘要
  - raw/articles/voltagent-llama-factory.md — VoltAgent 教程文章
- 创建页面:
  - entities/llama-factory.md — LLaMA Factory 框架详解
    - 框架架构 (Model Loader, Data Worker, Trainer)
    - 支持的高效训练技术 (LoRA, QLoRA, GaLore, 量化等)
    - 100+ 模型支持列表
    - 2025年新增功能 (OFT/OFTv2, 新优化器, SGLang)
    - 与其他框架对比 (Axolotl, TRL, Unsloth)
- 更新:
  - index.md — 添加 LLaMA Factory 到 Frameworks 和 Papers 部分

## [2026-04-23] ingest | 经典微调论文批量导入
- 来源: arXiv 论文摘要
- 导入论文 (7篇):
  1. raw/papers/arxiv-2106.09685.md — LoRA: Low-Rank Adaptation (Hu et al., 2021)
  2. raw/papers/arxiv-2305.14314.md — QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)
  3. raw/papers/arxiv-2203.02155.md — InstructGPT / RLHF (Ouyang et al., 2022)
  4. raw/papers/arxiv-2305.18290.md — DPO: Direct Preference Optimization (Rafailov et al., 2023)
  5. raw/papers/arxiv-2103.10385.md — P-Tuning: GPT Understands, Too (Liu et al., 2021)
  6. raw/papers/arxiv-2101.00190.md — Prefix-Tuning (Li & Liang, 2021)
  7. raw/papers/arxiv-1902.00751.md — Adapter: Parameter-Efficient Transfer Learning (Houlsby et al., 2019)
- 创建方法详解页面 (5篇):
  1. concepts/lora.md — LoRA 低秩适配详解
  2. concepts/qlora.md — QLoRA 量化LoRA详解
  3. concepts/rlhf.md — RLHF 人类反馈强化学习详解
  4. concepts/dpo.md — DPO 直接偏好优化详解
  5. concepts/adapter.md — Adapter 参数高效迁移详解
  6. concepts/p-tuning.md — P-Tuning 连续提示详解
  7. concepts/prefix-tuning.md — Prefix-Tuning 前缀微调详解
- 创建对比页面 (2篇):
  1. comparisons/peft-methods-comparison.md — PEFT 方法全面对比
  2. comparisons/rlhf-vs-dpo.md — RLHF vs DPO 对比
- 更新:
  - index.md — 添加所有新页面条目

## [2026-04-23] update | 下载论文PDF文件
- 为8篇论文下载完整PDF文件到 raw/papers/
- LoRA: lora-2106.09685.pdf (1.6M)
- QLoRA: qlora-2305.14314.pdf (1.1M)
- InstructGPT: instructgpt-2203.02155.pdf (1.8M)
- DPO: dpo-2305.18290.pdf (1.3M)
- P-Tuning: ptuning-2103.10385.pdf (1.6M)
- Prefix-Tuning: prefix-tuning-2101.00190.pdf (1.5M)
- Adapter: adapter-1902.00751.pdf (853K)
- LLaMA Factory: llamafactory-2403.13372.pdf (422K)
- 总计约10MB，存放于 raw/papers/ 目录

## [2026-04-23] update | 添加PDF引用链接
- 更新 raw/papers/ 下8篇论文源文件
- 每篇论文摘要页现可通过 `[[pdf文件名]]` wikilink 打开完整PDF

## [2026-04-23] create | 知识库大规模扩展 — 概念、实体、对比页
- 新增5个实体页: axolotl, trl, unsloth, deepspeed, bitsandbytes
- 新增8个概念页: dora, rslora, ppo, grpo, kto, gguf, gptq, awq
- 新增3个对比页: rl-methods-comparison, quantization-comparison, frameworks-comparison
- 更新 index.md: 总页面从12增至28
- 分类重组: Methods→Concepts, Frameworks→Entities

## [2026-04-23] ingest | 模型微调实际用例摄入
- 新增2个模型案例实体页: deepseek-r1, llama2-alignment
- 新增3个实践指南概念页: domain-finetuning, qlora-practice, finetuning-recipes
- DeepSeek-R1: GRPO纯RL训练全流程 + 蒸馏实验 + TRL/LLaMA Factory复现代码
- Llama 2 对齐: 5轮迭代RLHF + Ghost Attention + Llama 3改进
- 领域微调: 医疗(MedPaLM/Meditron)/代码(CodeLlama)/法律(ChatLaw)/金融(FinGPT)
- QLoRA实战: 硬件要求+配置详解+Python代码+常见问题排查
- 微调配方: 6个即用配方(中文对话/代码助手/DPO/GRPO/多模态/LoRA合并)
- 更新 index.md: 总页面从28增至34

## [2026-04-23] ingest | 法律领域微调专题
- 新增 concepts/legal-finetuning.md (15.8KB)
- 覆盖7个国内开源法律模型: ChatLaw, Lawyer LLaMA, LawGPT, 智海-录问, LaWGPT, HanFei, 天独
- 覆盖4个国际模型: SaulLM-7B, LegalBERT, LegalLED, CaseLawBERT
- 4个中文数据集: Chinese-Law-Dataset, LawGPT-Zh, LegalQA, CrimeKgAssitant
- 3个英文数据集: LegalBench, MultiLegal Pile, CaseHOLD
- 3套微调配置: ChatLaw风格, Lawyer LLaMA, 法律推理CoT
- RAG + 微调组合架构: 检索层(法条/判例/知识库) + 生成层(微调模型)
|- 法条向量库构建代码 + 自动评估脚本

## [2026-05-13] ingest | Hello-Agents Agentic-RL 教程
- 来源: Datawhale hello-agents 第11章
- 笔记: raw/papers/hello-agents-agentic-rl.md
- 内容: Agentic RL 完整训练流程（SFT + GRPO）
- 关键词: GRPO, PPO, SFT, LoRA, 奖励函数设计, 分布式训练
- 关联: [[concepts/grpo]], [[concepts/ppo]], [[concepts/lora]], [[concepts/rlhf]], [[entities/trl]]
