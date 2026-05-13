---
title: 法律领域微调实践
created: 2026-04-23
updated: 2026-04-23
type: concept
tags: [concept, legal, law, finetuning, chinese, dataset]
sources: []
---

# 法律领域微调实践

## 一句话总结

法律微调的核心是准确性 + 可追溯性，通常采用 RAG + 微调组合而非纯微调，确保法条引用准确无误。

## 开源法律大模型

### 国内开源模型

| 模型           | 基座             | 特点           | 开源地址                           |
| ------------ | -------------- | ------------ | ------------------------------ |
| ChatLaw      | Baichuan/Llama | 北大出品，判例+法条   | GitHub: PKU-YuanGroup/ChatLaw  |
| Lawyer LLaMA | Llama 2        | 中文法律指令微调     | GitHub: AndrewZhe/lawyer-llama |
| 蒲公英/LawGPT   | ChatGLM        | 中文法律问答       | GitHub: polaris-wang/LawGPT    |
| 智海-录问        | GLM            | 浙大+阿里，法律知识图谱 | GitHub: zjunlp/KnowLM          |
| 法言/LaWGPT    | Llama          | 法律领域持续预训练    | GitHub: Liu-Huan/LaWGPT        |
| 天独           | Qwen           | 天同律师事务所出品    | 暂未完全开源                         |
| HanFei       | Baichuan       | 韩非子，中文法律理解   | GitHub: SSENDOLyer/hanfei      |

### 国际开源模型

| 模型 | 基座 | 特点 | 开源地址 |
|------|------|------|----------|
| SaulLM-7B | Llama 2 | 法国法律LLM，多语言 | HuggingFace: Equall/SaulLM-7B |
| LegalBERT | BERT | 法律文本理解 | HuggingFace: nlpaueb/legal-bert |
| LegalLED | LED | 法律文档摘要 | HuggingFace: nz/legal-led |
| CaseLawBERT | BERT | 判例法理解 | HuggingFace: zgkar/CaseLawBERT |

## 法律微调数据集

### 中文法律数据集

#### 1. Chinese-Law-Dataset

```yaml
名称: Chinese-Law-Dataset
来源: GitHub: liuhuanzhang/Chinese-Law-Dataset
规模:
  法条: 10万+ 条 (现行法律全文)
  判例: 500万+ 份裁判文书摘要
  问答: 10万+ 法律问答对
格式:
  法条JSON:
    {
      "law_name": "中华人民共和国民法典",
      "article_num": "第一百四十三条",
      "content": "具备下列条件的民事法律行为有效...",
      "category": "民法"
    }
  判例JSON:
    {
      "case_id": "2023京0105民初12345号",
      "case_type": "民事案件",
      "facts": "原告与被告...",
      "judgment": "本院判决...",
      "relevant_laws": ["民法典第143条", "合同法第107条"]
    }
```

#### 2. LawGPT-Zh

```yaml
名称: LawGPT-Zh (蒲公英数据集)
来源: GitHub: polaris-wang/LawGPT
规模: 20万+ 中文法律问答
格式: Alpaca 格式
{
  "instruction": "根据以下案情，分析适用的法律条款",
  "input": "张三向李四借款10万元，约定一年归还，未签书面合同...",
  "output": "根据《民法典》第六百七十九条，自然人之间的借款合同..."
}
领域覆盖:
  - 民商法 (40%)
  - 刑法 (20%)
  - 行政法 (15%)
  - 劳动法 (10%)
  - 婚姻家庭 (10%)
  - 知识产权 (5%)
```

#### 3. LegalQA

```yaml
名称: LegalQA (法律问答)
来源: HuggingFace:蒜片网/legal-qa-zh
规模: 5万+ 专业律师回答
特点:
  - 真实用户提问
  - 专业律师回答
  - 包含追问对话
格式:
{
  "question": "公司拖欠工资三个月，我该如何维权？",
  "answer": "您可以采取以下途径维权：\n1. 向劳动监察大队投诉...",
  "lawyer_info": "张律师，执业10年，劳动法专业",
  "category": "劳动法"
}
```

#### 4. CrimeKgAssitant

```yaml
名称: CrimeKgAssitant (刑法知识图谱)
来源: GitHub: confession/Run-Lawyer-LLM
规模:
  罪名: 400+ 个刑法罪名
  知识三元组: 5万+
  案例问答: 2万+
格式:
{
  "crime": "诈骗罪",
  "definition": "以非法占有为目的，用虚构事实...",
  "constitutive_elements": ["主观故意", "虚构事实", "骗取财物"],
  "sentencing_guide": "数额较大处三年以下...",
  "related_crimes": ["合同诈骗罪", "集资诈骗罪"]
}
```

### 英文法律数据集

#### 1. LegalBench

```yaml
名称: LegalBench
来源: GitHub: HazyResearch/legalbench
规模: 162个任务，6K+ 测试样本
任务类型:
  - Issue Spotting: 识别法律问题
  - Rule Recitation: 复述法律规则
  - Rule Application: 应用规则到案例
  - Conclusion: 得出结论
用途: 法律模型评测基准
```

#### 2. MultiLegal Pile

```yaml
名称: MultiLegal Pile
来源: HuggingFace
规模: 689GB 多语言法律文本
语言: 英语、德语、法语、意大利语等
内容:
  - 法律条文
  - 法庭判决
  - 法律期刊
  - 合同文书
用途: 法律领域持续预训练
```

#### 3. CaseHOLD

```yaml
名称: CaseHOLD
来源: Stanford
规模: 53K+ 法案引用预测样本
任务: 给定法律问题，预测应引用哪个法条
格式:
{
  "context": "合同纠纷案件...",
  "holding": "《合同法》第107条",
  "choices": ["A. 合同法第107条", "B. 民法典第119条", ...]
}
```

## 微调配置示例

### ChatLaw 微调配置

```yaml
# ChatLaw 风格法律模型微调
基座: Baichuan2-13B-Chat / Qwen2.5-14B
数据混合:
  法条预训练: 100%
  指令微调:
    - 法条问答: 40%
    - 案例分析: 30%
    - 法律推理: 20%
    - 文书生成: 10%

训练配置:
  stage: sft
  finetuning_type: lora
  lora_rank: 128            # 法律需要更高秩
  lora_alpha: 256
  lora_target: all
  
  learning_rate: 2e-5       # 法律数据敏感，用较低学习率
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_length: 4096          # 法律文书较长
  
  # 数据增强
  augmentation: true
  augmentation_methods:
    - paraphrase            # 同义改写
    - back_translation      # 回译增强

评估:
  - LegalBench 中文版
  - 司法考试真题
  - 法条引用准确率
```

### Lawyer LLaMA 微调配置

```yaml
# Lawyer LLaMA 配置
基座: Llama-2-13B
数据:
  指令数据: 50K 中文法律问答
  法条数据: 全部现行法条作为知识增强
  
训练流程:
  阶段1: 持续预训练
    数据: 法律条文 + 判例文书 (10B tokens)
    learning_rate: 1e-5
    epochs: 1
  
  阶段2: 指令微调
    数据: 50K 法律问答
    learning_rate: 3e-5
    lora_rank: 64

特殊处理:
  - 法条编号作为特殊token
  - 法律术语词典注入
  - 罪名实体识别
```

### 法律推理微调

```yaml
# 法律推理能力微调 (类似 CoT)
基座: Qwen2.5-7B-Instruct

数据格式 (法律推理链):
{
  "instruction": "分析以下案件的法律适用",
  "input": "张三向李四借款10万元，未约定利息...",
  "output": "
    ## 案件定性
    本案属于民间借贷纠纷。
    
    ## 适用法律
    1. 《民法典》第六百七十九条：自然人之间的借款合同...
    2. 《民法典》第六百八十条：借款合同对支付利息...
    
    ## 法律分析
    - 借贷关系成立：张三向李四借款，双方意思表示真实...
    - 未约定利息的处理：视为不支付利息...
    
    ## 结论
    李四有权要求张三归还本金10万元，但无权主张利息。
  "
}

训练配置:
  stage: sft
  lora_rank: 64
  max_length: 4096
  learning_rate: 3e-5
  num_train_epochs: 5       # 推理数据少，多训练几轮
  
  # 保持格式一致性
  temperature: 0.3          # 低温度减少幻觉
```

## RAG + 微调 组合方案

### 为什么法律需要 RAG

```yaml
纯微调的问题:
  1. 法条幻觉: 模型可能编造不存在的法条
  2. 过时问题: 法律更新频繁，微调后无法及时更新
  3. 引用不准确: 无法精确引用法条出处
  4. 责任问题: 无法追溯回答来源

RAG + 微调 优势:
  1. 准确引用: 回答基于检索到的真实法条
  2. 可追溯: 用户可验证法条来源
  3. 易更新: 法律更新只需更新检索库
  4. 降低幻觉: 检索结果约束生成
```

### RAG + 微调 架构

```
用户提问
    │
    ▼
┌─────────────────────────────────────┐
│  检索层                      │
│  ┌─────────┐  ┌─────────┐  ┌──────┐ │
│  │法条向量库│  │判例向量库│  │知识库│ │
│  └────┬────┘  └────┬────┘  └──┬───┘ │
│       └────────────┼──────────┘     │
│                    ▼                │
│            检索结果 Top-K           │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│  生成层            │
│  ┌───────────────────────────────┐  │
│  │ Prompt: 检索结果 + 用户问题   │  │
│  │ → 微调模型生成回答            │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
                    │
                    ▼
              结构化回答 (含法条引用)
```

### RAG 实现代码

```python
# 法律 RAG + 微调 示例
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载法条向量库
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh"  # 中文向量模型
)
law_vectorstore = FAISS.load_local(
    "./law_vectors", 
    embeddings,
    allow_dangerous_deserialization=True
)

# 2. 加载微调模型
model = AutoModelForCausalLM.from_pretrained(
    "./lawyer-model-finetuned",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./lawyer-model-finetuned")

# 3. RAG 问答
def legal_qa(question: str):
    # 检索相关法条
    docs = law_vectorstore.similarity_search(question, k=5)
    context = "\n\n".join([
        f"【{d.metadata.get('law_name')} 第{d.metadata.get('article')}条】\n{d.page_content}"
        for d in docs
    ])
    
    # 构建提示
    prompt = f"""你是一个专业的法律顾问。请根据以下法条回答用户问题，并在回答中引用相关法条。

相关法条：
{context}

用户问题：{question}

请给出专业的法律分析，并明确引用相关法条：""
    
    # 生成回答
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.3)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 法条向量库构建

```python
# 构建法条向量库
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 加载法条数据
with open("./chinese_laws.json") as f:
    laws = json.load(f)

# 转换为 Document
documents = []
for law in laws:
    doc = Document(
        page_content=law["content"],
        metadata={
            "law_name": law["law_name"],
            "article": law["article_num"],
            "category": law["category"],
            "source": law.get("source", "official")
        }
    )
    documents.append(doc)

# 构建向量库
vectorstore = FAISS.from_documents(
    documents,
    embeddings
)
vectorstore.save_local("./law_vectors")
```

## 法律微调数据标注

### 数据标注规范

```yaml
法律数据标注要求:
  1. 法条引用: 每条回答必须引用具体法条
  2. 准确性: 人工审核法条引用正确性
  3. 完整性: 引用完整法条号，如"《民法典》第一百四十三条"
  4. 时效性: 标注数据来源日期，法律可能有变化

标注流程:
  1. 律师撰写答案 → 2. 法条核对 → 3. 格式审核 → 4. 入库
```

### 数据格式模板

```json
{
  "id": "legal_qa_001",
  "instruction": "分析以下案件的法律关系和适用法条",
  "input": "甲方与乙方签订房屋买卖合同，约定总价200万元...",
  "output": "## 案件法律关系分析\n\n本案涉及房屋买卖合同纠纷，法律关系如下：\n\n### 1. 合同效力\n根据《民法典》第二百一十五条，当事人之间订立有关设立、变更、转让和消灭不动产物权的合同，除法律另有规定或者当事人另有约定外，自合同成立时生效；未办理物权登记的，不影响合同效力。\n\n本案中，虽然双方未办理过户登记，但买卖合同已生效，乙方有权要求甲方履行过户义务。\n\n### 2. 违约责任\n根据《民法典》第五百七十七条，当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。\n\n甲方拒绝过户的行为构成违约，应承担继续履行或赔偿损失的责任。",
  "metadata": {
    "category": "合同法",
    "laws_cited": ["民法典第215条", "民法典第577条"],
    "difficulty": "medium",
    "verified_by": "张律师",
    "verified_date": "2024-03-15"
  }
}
```

## 评估指标

### 法律模型评估

```yaml
评估维度:
  1. 法条准确率: 引用的法条是否正确存在
  2. 适用准确率: 引用的法条是否适用于该案情
  3. 推理完整度: 法律推理过程是否完整
  4. 回答质量: 回答是否专业、易懂

评测数据集:
  - 司法考试真题
  - LegalBend 中文版
  - 真实案例测试集
  - 律师人工评估
```

### 自动评估脚本

```python
# 法条引用准确性检查
import re

def check_law_citations(response, law_database):
    """检查回答中的法条引用是否正确"""
    # 提取所有法条引用
    pattern = r'《([^》]+)》第([一二三四五六七八九十百千零\d]+)条'
    citations = re.findall(pattern, response)
    
    results = []
    for law_name, article in citations:
        # 查询法条是否存在
        if law_name in law_database:
            if article in law_database[law_name]:
                results.append({
                    "citation": f"《{law_name}》第{article}条",
                    "status": "正确",
                    "content": law_database[law_name][article][:50] + "..."
                })
            else:
                results.append({
                    "citation": f"《{law_name}》第{article}条",
                    "status": "错误: 该法条不存在"
                })
        else:
            results.append({
                "citation": f"《{law_name}》第{article}条",
                "status": "错误: 该法律法规不存在"
            })
    
    return results
```

## 经验总结

### 法律微调最佳实践

| 要点 | 说明 |
|------|------|
| RAG 是必需的 | 纯微调无法避免幻觉，必须结合检索 |
| 低温度生成 | 法律回答要求准确，温度建议 0.1-0.3 |
| 法条特殊处理 | 法条编号作为特殊token，便于精确识别 |
| 定期更新 | 法律更新时，需及时更新检索库 |
| 人工审核 | 所有训练数据必须由律师审核 |
| 引用格式化 | 统一法条引用格式，如《民法典》第X条 |

### 常见问题

| 问题 | 解决方案 |
|------|----------|
| 编造不存在的法条 | 必须 RAG，仅用检索结果生成 |
| 引用过时法条 | 标注数据时效，定期更新检索库 |
| 法条文号错误 | 法条编号作为特殊 token 训练 |
| 回答不专业 | 增加律师标注数据 |
| 无法追溯来源 | RAG 架构天然支持追溯 |

## 相关内容

- [[domain-finetuning|领域微调实践案例]] — 医疗/法律/代码/金融
- [[qlora-practice|QLoRA 实战配置]] — 低成本微调方案
- [[finetuning-recipes|开源微调配方集]] — 即用配方
