---
title: GPTQ: Generative Pre-trained Transformer Quantization
created: 2026-04-23
updated: 2026-04-23
type: concept
tags: [gptq, quantization, inference, gpu-inference, 4bit]
sources: []
paper: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
arxiv_id: "2210.17323"
---

# GPTQ: 生成式预训练变换器量化

## 一句话总结
基于近似二阶信息的训练后量化方法，4-bit量化几乎无精度损失，是 GPU 推理量化的主流选择之一。

## 核心原理

GPTQ 的核心思想：逐层量化权重时，用 Hessian 矩阵的近似来补偿量化误差。

### OBQ (Optimal Brain Quantization)

GPTQ 基于最优脑量化 (OBQ) 方法：

1. **逐行量化**：每次量化一个权重，立即调整未量化权重补偿误差
2. **Hessian 引导**：用量化误差的 Hessian 矩阵决定最优调整方向
3. **贪心顺序**：按量化误差从小到大的顺序量化

### GPTQ 的三个加速

OBQ 的计算复杂度是 $O(d_{row} \cdot d_{col}^3)$，GPTQ 通过三个优化降到 $O(d_{col}^3)$：

1. **批量处理**：同时量化一整行，而非逐个权重
2. **计算复用**：利用 Cholesky 分解避免重复计算
3. **固定顺序**：按自然顺序处理，省去贪心排序

### 算法流程

```
对于每一层:
1. 收集校准数据，计算 Hessian H⁻¹
2. Cholesky 分解 H⁻¹ = L·Lᵀ
3. 逐行量化:
   For each row w:
     a. 量化: w_q = quantize(w)
     b. 计算误差: δ = w - w_q
     c. 用 Cholesky 调整未量化列: w[未量化] -= δ / H⁻¹[已量化, 未量化]
4. 量化后的权重存储为 INT4
```

## 量化配置

### 精度选项

| 配置 | 位数 | 组大小 | 说明 |
|------|------|--------|------|
| `4bit-128g` | 4-bit | 128 | **最常用**，平衡好 |
| `4bit-64g` | 4-bit | 64 | 更精确，文件略大 |
| `4bit-32g` | 4-bit | 32 | 最精确，文件最大 |
| `3bit-128g` | 3-bit | 128 | 极限压缩 |
| `8bit` | 8-bit | - | 几乎无损 |

组大小越小，精度越高，但文件越大。

## 量化流程

### 使用 AutoGPTQ
```bash
pip install auto-gptq

# Python 量化
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 配置
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,        # 按激活大小排序
    damp_percent=0.01,    # Hessian 阻尼
)

# 加载模型
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b", quantize_config
)

# 校准数据 (128-256 条)
calib_data = load_calibration_data()

# 量化
model.quantize(calib_data)

# 保存
model.save_quantized("Llama-2-7b-GPTQ-4bit")
```

### 使用 HuggingFace 推理
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto",
    # GPTQ 自动识别
)
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-GPTQ")
```

## 显存对比 (Llama-2-7B)

| 方式 | 模型大小 | 推理显存 |
|------|----------|----------|
| FP16 | ~14GB | ~16GB |
| GPTQ-8bit | ~7GB | ~9GB |
| GPTQ-4bit | ~3.9GB | ~6GB |
| GPTQ-3bit | ~3.0GB | ~5GB |

## 精度对比

| 方法 | 位数 | Perplexity (Wiki2) | 相对 FP16 |
|------|------|-------------------|-----------|
| FP16 | 16 | 5.47 | 基线 |
| GPTQ-8bit | 8 | 5.50 | +0.5% |
| GPTQ-4bit-128g | 4 | 5.63 | +2.9% |
| GPTQ-3bit-128g | 3 | 6.27 | +14.6% |
| RTN-4bit | 4 | 6.79 | +24.1% |

GPTQ 4-bit 仅比 FP16 差 2.9%，远优于简单取整 (RTN)。

## 与其他量化方案对比

| 特性 | GPTQ | [[gguf\|GGUF]] | [[awq\|AWQ]] | bitsandbytes |
|------|------|------|------|-------------|
| 量化方式 | 训练后 | 训练后 | 训练后 | 动态 |
| 需要校准数据 | ✅ (128条) | ❌ | ✅ (128条) | ❌ |
| 量化时间 | 慢 (10-30min) | 快 (分钟) | 中 (5-10min) | 即时 |
| GPU 推理 | ✅ **最佳** | 部分 | ✅ | ✅ |
| CPU 推理 | ❌ | ✅ **最佳** | ❌ | ❌ |
| 精度 | **最高** | 好 | 好 | 好 |
| 支持训练 | ❌ | ❌ | ❌ | ✅ |

## 优缺点

| 优点 | 缺点 |
|------|------|
| 4-bit 精度损失极小 | 量化过程慢 |
| GPU 推理效率高 | 不支持 CPU |
| 校准数据需求少 | 不支持训练 |
| 生态支持好 (vLLM等) | 模型兼容性偶有问题 |

## 适用场景

- GPU 服务器推理部署
- 需要最高量化精度
- 配合 vLLM / ExLlamaV2 高吞吐推理
- 多模型批量服务

## 相关资源

- AutoGPTQ: https://github.com/AutoGPTQ/AutoGPTQ
- GPTQ 论文: https://arxiv.org/abs/2210.17323
- 预量化模型: https://huggingface.co/TheBloke

## 相关内容

- [[awq|AWQ]] — 激活感知量化
- [[gguf|GGUF]] — CPU推理格式
- [[qlora|QLoRA]] — 训练时量化
- [[bitsandbytes]] — 动态量化库
