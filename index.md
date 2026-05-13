# LLM Fine-tuning Wiki Index

> 大模型微调知识库内容目录。每个页面一行：链接 + 简介。
> 首先阅读此文件以找到相关页面。
> 最后更新: 2026-05-13 | 总页面: 35

## Papers (论文笔记)
<!-- 按发表时间倒序 -->

| 论文                            | 年份                                                 | 领域   |      |
| ----------------------------- | -------------------------------------------------- | ---- | ---- |
| [[raw/papers/arxiv-2403.13372 | LlamaFactory: Unified Efficient Fine-Tuning]]      | 2024 | 框架   |
| [[raw/papers/arxiv-2305.18290 | DPO: Direct Preference Optimization]]              | 2023 | 对齐   |
| [[raw/papers/arxiv-2305.14314 | QLoRA: Efficient Finetuning of Quantized LLMs]]    | 2023 | 量化微调 |
| [[raw/papers/arxiv-2203.02155 | InstructGPT: Training LMs to Follow Instructions]] | 2022 | RLHF |
| [[raw/papers/arxiv-2106.09685 | LoRA: Low-Rank Adaptation]]                        | 2021 | PEFT |
| [[raw/papers/arxiv-2103.10385 | P-Tuning: GPT Understands, Too]]                   | 2021 | 提示微调 |
| [[raw/papers/arxiv-2101.00190 | Prefix-Tuning: Optimizing Continuous Prompts]]     | 2021 | 提示微调 |
| [[raw/papers/arxiv-1902.00751 | Adapter: Parameter-Efficient Transfer Learning]]   | 2019 | PEFT |
| [[raw/papers/hello-agents-agentic-rl | Hello-Agents: Agentic-RL 训练教程]]          | 2025 | GRPO训练 |

## Articles (文章教程)
- [[raw/articles/voltagent-llama-factory|VoltAgent: LLaMA Factory 介绍]] — 2025年最新功能更新

## Concepts (方法详解)
<!-- 按方法类型分组 -->

### PEFT 方法
- [[lora|LoRA]] — 低秩适配，冻结权重注入可训练低秩矩阵
- [[qlora|QLoRA]] — 4-bit量化+LoRA，单卡微调65B模型
- [[adapter|Adapter]] — 层间插入瓶颈模块，仅3.6%参数
- [[prefix-tuning|Prefix-Tuning]] — 每层添加可训练前缀向量
- [[p-tuning|P-Tuning]] — 可训练连续提示嵌入替代离散提示
- [[dora|DoRA]] — 权重分解LoRA，将权重拆分为幅度和方向分别适配
- [[rslora|RSLoRA]] — 秩稳定LoRA，用1/√r替代1/r缩放提升高秩效果

### RL 对齐方法
- [[rlhf|RLHF]] — 人类反馈强化学习，三阶段对齐流程
- [[dpo|DPO]] — 直接偏好优化，绕过奖励模型简化RLHF
- [[ppo|PPO]] — 近端策略优化，RLHF的核心策略优化算法
- [[grpo|GRPO]] — 组相对策略优化，DeepSeek-R1的规则奖励方法
- [[kto|KTO]] — Kahneman-Tversky优化，只需二元标注的对齐方法

### 量化方法
- [[gguf|GGUF]] — llama.cpp量化格式，CPU推理首选
- [[gptq|GPTQ]] — GPU后训练量化，基于Hessian的逐层量化
- [[awq|AWQ]] — 激活感知量化，保护重要通道提升精度

### 实践指南
- [[domain-finetuning|领域微调实践案例]] — 医疗/法律/代码/金融四领域微调详解
- [[qlora-practice|QLoRA 实战配置]] — QLoRA 完整配置指南与常见问题
- [[finetuning-recipes|开源微调配方集]] — 6个社区验证的微调配方（对话/代码/DPO/GRPO/多模态/合并）
- [[legal-finetuning|法律领域微调实践]] — 7个开源法律模型+8个数据集+RAG组合方案+评估指标

## Entities (实体)
<!-- 框架、工具、库、模型 -->

### 训练框架
- [[llama-factory|LLaMA Factory]] — 统一微调框架，支持100+模型，零代码Web UI
- [[axolotl|Axolotl]] — 高级微调框架，YAML配置驱动，社区配方丰富
- [[trl|TRL]] — HuggingFace强化学习库，最新方法第一时间支持
- [[unsloth|Unsloth]] — 极速微调框架，2-5x加速，显存减半
- [[deepspeed|DeepSpeed]] — 微软分布式训练引擎，ZeRO内存优化

### 工具库
- [[bitsandbytes|bitsandbytes]] — 动态量化库，NF4/Int8训练时量化

### 模型案例
- [[deepseek-r1|DeepSeek-R1]] — GRPO纯RL训练涌现推理，多阶段SFT→GRPO→DPO
- [[llama2-alignment|Llama 2 对齐训练]] — 工业级RLHF标杆，5轮迭代对齐+Ghost Attention

## Comparisons (对比分析)
- [[peft-methods-comparison|PEFT 方法对比]] — LoRA/QLoRA/Adapter/Prefix-Tuning/P-Tuning 全面对比
- [[rlhf-vs-dpo|RLHF vs DPO 对比]] — 两种对齐方法的流程、效果和资源对比
- [[rl-methods-comparison|RL 对齐方法对比]] — PPO/RLHF/DPO/KTO/GRPO 五种方法全面对比
- [[quantization-comparison|量化方法对比]] — GGUF/GPTQ/AWQ/bitsandbytes 四种量化方案对比
- [[frameworks-comparison|训练框架对比]] — LLaMA Factory/Axolotl/TRL/Unsloth/DeepSpeed 五大框架对比

## Queries (查询记录)
<!-- 有价值的查询结果存档 -->

## Benchmarks (评测基准)
<!-- MMLU, HumanEval 等 -->

## Datasets (数据集)
<!-- 训练数据资源 -->
