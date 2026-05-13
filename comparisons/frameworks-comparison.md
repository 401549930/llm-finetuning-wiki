---
title: 训练框架对比
created: 2026-04-23
updated: 2026-04-23
type: comparison
tags: [comparison, framework, llama-factory, axolotl, trl, unsloth, deepspeed]
sources: []
---

# 训练框架对比

## 对比概览

| 框架 | 定位 | 易用性 | 灵活性 | 速度 | 社区 |
|------|------|--------|--------|------|------|
| [[llama-factory\|LLaMA Factory]] | 一站式 | ★★★★★ | ★★★ | ★★★ | ★★★★★ |
| [[axolotl\|Axolotl]] | 高级定制 | ★★★ | ★★★★★ | ★★★★ | ★★★★ |
| [[trl\|TRL]] | 研究/库 | ★★★ | ★★★★★ | ★★★ | ★★★★ |
| [[unsloth\|Unsloth]] | 极速微调 | ★★★★★ | ★★ | ★★★★★ | ★★★★ |
| [[deepspeed\|DeepSpeed]] | 分布式 | ★★ | ★★★ | ★★★★★ | ★★★★ |

## 详细对比

### 功能矩阵

| 功能 | LLaMA Factory | Axolotl | TRL | Unsloth | DeepSpeed |
|------|:---:|:---:|:---:|:---:|:---:|
| SFT | ✅ | ✅ | ✅ | ✅ | ✅ |
| LoRA/QLoRA | ✅ | ✅ | ✅ | ✅ | ✅ |
| DPO | ✅ | ✅ | ✅ | ✅ | ❌ |
| RLHF/PPO | ✅ | ✅ | ✅ | ❌ | ✅ |
| KTO | ✅ | ✅ | ✅ | ❌ | ❌ |
| GRPO | ✅ | ✅ | ✅ | ❌ | ❌ |
| 全参数微调 | ✅ | ✅ | ✅ | ❌ | ✅ |
| 多模态 | ✅ | ✅ | ❌ | ❌ | ❌ |
| 分布式训练 | DeepSpeed/FSDP | DeepSpeed/FSDP | 需配合 | ❌ | ✅ 原生 |
| 模型合并 | ✅ | ✅ | ❌ | ❌ | ❌ |
| Web UI | ✅ | ❌ | ❌ | ❌ | ❌ |
| 量化推理 | ✅ | ❌ | ❌ | ❌ | ❌ |

### 配置方式

| 框架 | 配置格式 | 学习曲线 |
|------|----------|----------|
| LLaMA Factory | YAML / CLI / Web UI | **最低** |
| Axolotl | YAML | 中等 |
| TRL | Python 脚本 | 较高 |
| Unsloth | Python 脚本 | 低 |
| DeepSpeed | JSON 配置 | **最高** |

### 性能对比 (Llama-2-7B, SFT, 单卡 A100)

| 框架 | 吞吐量 (token/s) | 显存占用 | 训练时间 (1M tokens) |
|------|------------------|----------|---------------------|
| 标准 HuggingFace | ~2000 | 16GB | ~8min |
| LLaMA Factory | ~2200 | 14GB | ~7.5min |
| Axolotl | ~2400 | 14GB | ~7min |
| TRL | ~2000 | 16GB | ~8min |
| **Unsloth** | **~5000** | **10GB** | **~3.5min** |
| DeepSpeed ZeRO-3 | ~2000 | 分散 | 取决于节点数 |

Unsloth 在单卡场景优势明显 (2-5x 加速)。

### 多卡/分布式支持

| 框架 | 单卡 | 多卡 | 多节点 | 策略 |
|------|------|------|--------|------|
| LLaMA Factory | ✅ | ✅ | ✅ | DeepSpeed/FSDP |
| Axolotl | ✅ | ✅ | ✅ | DeepSpeed/FSDP |
| TRL | ✅ | 需配合 | 需配合 | 需 Accelerate |
| Unsloth | ✅ | **仅多卡** | ❌ | 有限 |
| DeepSpeed | ✅ | ✅ | ✅ | ZeRO-1/2/3 |

### 支持模型范围

| 框架 | 支持模型数 | 自定义模型 |
|------|-----------|-----------|
| LLaMA Factory | 60+ | 配置即可 |
| Axolotl | 30+ | 需适配 |
| TRL | 任意 HF | 直接加载 |
| Unsloth | 15+ | 部分支持 |
| DeepSpeed | 任意 | 直接加载 |

## 按场景选择

### 新手入门

| 推荐 | 理由 |
|------|------|
| **LLaMA Factory** | Web UI + 预设模板，零代码上手 |

### 快速实验

| 推荐 | 理由 |
|------|------|
| **Unsloth** | 2-5x加速，适合快速迭代 |
| LLaMA Factory | 一键启动，模板丰富 |

### 研究/自定义

| 推荐 | 理由 |
|------|------|
| **TRL** | 灵活的Python API，方便魔改 |
| Axolotl | 高级配置 + 社区配方 |

### 大规模训练

| 推荐 | 理由 |
|------|------|
| **DeepSpeed** | 原生分布式，ZeRO优化 |
| Axolotl + DeepSpeed | 配置驱动 + 分布式 |

### 生产部署

| 推荐 | 理由 |
|------|------|
| **LLaMA Factory** | 端到端：训练→导出→部署 |
| Axolotl | 配置可复现 |

## 组合使用

### 最佳实践组合

```
训练框架 + 加速库组合:

入门:     LLaMA Factory (WebUI) + DeepSpeed
快速实验: Unsloth + QLoRA
研究:     TRL + Accelerate + DeepSpeed
生产:     Axolotl + DeepSpeed ZeRO-3
微调+部署: LLaMA Factory → GGUF → Ollama
```

### 量化 + 框架组合

| 组合 | 显存 (7B) | 速度 | 质量 |
|------|-----------|------|------|
| Unsloth + 4bit | **10GB** | ★★★★★ | ★★★★ |
| LLaMA Factory + QLoRA | 12GB | ★★★ | ★★★★ |
| Axolotl + QLoRA | 12GB | ★★★★ | ★★★★★ |
| TRL + BnB | 12GB | ★★★ | ★★★★ |

## 生态系统

### LLaMA Factory 生态
- 模型仓库: ModelScope + HuggingFace
- 数据集: 10+ 预置数据集
- 社区: GitHub 30K+ Stars, 中文社区活跃

### Axolotl 生态
- 配方库: 社区共享 YAML 配方
- 集成: DeepSpeed, FSDP, Flash Attention
- 部署: 导出标准 HF 格式

### TRL 生态
- HuggingFace 亲儿子: 与 Transformers 深度集成
- 研究: 最新方法第一时间支持
- 教程: 丰富的官方 Colab

### Unsloth 生态
- 速度: 核心卖点，2-5x 加速
- 显存: 比标准方案省 50-70%
- 限制: 模型支持范围有限

### DeepSpeed 生态
- 微软: 企业级支持
- ZeRO: 三级内存优化
- 通用: 底层引擎，被所有框架调用

## 迁移路径

```
新手路径: LLaMA Factory → Axolotl → DeepSpeed
研究路径: TRL → TRL + DeepSpeed
快速路径: Unsloth (保持即可)
```

从 LLaMA Factory 入门，需要更多自定义时转向 Axolotl，需要大规模分布式时加入 DeepSpeed。

## 相关内容

- [[llama-factory|LLaMA Factory]] — 一站式微调框架
- [[axolotl|Axolotl]] — 高级微调框架
- [[trl|TRL]] — 强化学习训练库
- [[unsloth|Unsloth]] — 极速微调框架
- [[deepspeed|DeepSpeed]] — 分布式训练引擎
- [[bitsandbytes]] — 量化库（常与框架组合）
- [[lora|LoRA]] — 常用微调方法
