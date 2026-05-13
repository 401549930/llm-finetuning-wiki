# LLM Fine-tuning Wiki

大模型微调方法与论文知识库。

## 知识库结构

```
llm-finetuning-wiki/
├── SCHEMA.md        # 领域规范、标签分类
├── index.md         # 内容目录
├── log.md           # 操作日志
├── raw/             # 原始材料（不可修改）
│   ├── articles/    # 网页文章
│   ├── papers/      # 论文原文
│   ├── transcripts/ # 转录文本
│   └── assets/      # 图片资源
├── entities/        # 实体页面（人物、机构、框架）
├── concepts/        # 概念页面（方法、理论）
├── comparisons/     # 对比分析
├── queries/         # 查询存档
├── _templates/      # 页面模板
└── scripts/         # 自动化脚本
```

## 覆盖领域

### 微调方法
- **PEFT**: LoRA, QLoRA, Adapter, Prefix Tuning, Prompt Tuning
- **RL**: RLHF, DPO, PPO, GRPO, KTO, SPO

### 训练框架
- Axolotl, TRL, Unsloth, LLaMA-Factory, DeepSpeed, FSDP

### 量化技术
- GGUF, GPTQ, AWQ, bitsandbytes

### 评测基准
- MMLU, HumanEval, MT-Bench, GSM8K

## 使用方式

### 添加论文
```
1. 将论文PDF/链接存入 raw/papers/
2. 告诉 Agent: "导入论文 [论文标题/链接]"
3. Agent 会生成论文笔记并更新索引
```

### 添加方法
```
1. 提供方法说明或相关文章
2. Agent 会创建方法详解页面
3. 自动关联相关内容
```

### 查询知识
```
直接提问，如：
- "LoRA 和 QLoRA 有什么区别？"
- "推荐一个适合7B模型的微调方案"
- "最新DPO论文有哪些进展？"
```

## 标签体系

详见 SCHEMA.md，主要分类：
- 微调方法: lora, qlora, peft, adapter...
- RL方法: rlhf, dpo, ppo, grpo...
- 训练框架: axolotl, trl, unsloth...
- 量化: gguf, gptq, awq...
- 内容类型: paper, method, comparison...

## 在 Obsidian 中打开

1. 打开 Obsidian
2. 选择 "打开文件夹作为仓库"
3. 选择 `llm-finetuning-wiki` 目录
4. 推荐安装插件：Dataview, Templater

## 维护原则

1. **原始材料不可修改**: raw/ 目录仅追加
2. **每次操作记录日志**: 更新 log.md
3. **新页面必须入索引**: 更新 index.md
4. **保持交叉链接**: 每页至少2个出链
5. **标签遵循分类**: 只使用 SCHEMA.md 中定义的标签
