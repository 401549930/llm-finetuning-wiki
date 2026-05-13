---
title: AWQ: Activation-aware Weight Quantization
created: 2026-04-23
updated: 2026-04-23
type: concept
tags: [awq, quantization, inference, gpu-inference, 4bit]
sources: []
paper: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
arxiv_id: "2306.00978"
---

# AWQ: 激活感知权重量化

## 一句话总结
保护"重要权重"的量化方法，通过激活幅度识别关键权重并给予更高精度，4-bit量化保持更好性能。

## 核心原理

AWQ 的核心洞察：**权重对模型性能的重要性不同**，应保护那些对激活贡献大的权重。

### 关键发现

1. **权重重要性不均**：部分权重对激活输出影响巨大
2. **激活幅度指示重要性**：激活幅度大的通道对应更重要权重
3. **保护重要权重**：对这些权重减少量化幅度

### 与 GPTQ 的区别

| 方面 | [[gptq\|GPTQ]] | AWQ |
|------|---------------|-----|
| 误差补偿 | Hessian 逆校正 | **通道缩放** |
| 重要权重 | 隐式 (Hessian) | **显式 (激活)** |
| 量化顺序 | 自然顺序 | **激活感知** |
| 实现复杂度 | 高 | **低** |

## 算法细节

### 1. 激活感知的权重保护

```
步骤:
1. 收集校准数据，计算各通道激活幅度
2. 识别重要通道 (激活幅度大的前1-5%)
3. 对应权重通道使用较小量化步长
4. 其余通道正常量化
```

### 2. 通道缩放

关键公式：
$$W_{scaled} = s \odot W$$
$$X_{scaled} = X / s$$

通过缩放因子 $s$ 放大重要通道的权重，间接减小其量化误差：
$$\text{量化误差} = \frac{\Delta}{s_{channel}}$$

$s$ 越大，该通道的相对量化误差越小。

### 3. 缩放因子优化

AWQ 通过网格搜索找到最优缩放因子：
$$s^* = \arg\min_s \text{Error}(Q(W \odot s) / s)$$

## 实现示例

### 使用 AutoAWQ
```bash
pip install autoawq

# 量化脚本
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 加载模型
model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# 配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",  # 或 GEMV
}

# 校准数据
calib_data = load_calibration_data()

# 量化
model.quantize(
    tokenizer,
    calib_data=calib_data,
    quant_config=quant_config,
)

# 保存
model.save_quantized("Llama-2-7b-AWQ")
```

### 推理
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_quantized("Llama-2-7b-AWQ")
tokenizer = AutoTokenizer.from_pretrained("Llama-2-7b-AWQ")

input_text = "Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()
outputs = model.generate(input_ids, max_length=50)
print(tokenizer.decode(outputs[0]))
```

## 精度对比

### Perplexity (Llama-2-7B, Wiki2)

| 方法 | 位数 | Perplexity | 相对 FP16 |
|------|------|------------|-----------|
| FP16 | 16 | 5.47 | 基线 |
| RTN-4bit | 4 | 6.79 | +24.1% |
| GPTQ-4bit | 4 | 5.63 | +2.9% |
| AWQ-4bit | 4 | **5.58** | **+2.0%** |
| AWQ-3bit | 3 | 5.95 | +8.8% |

AWQ 4-bit 精度略优于 GPTQ。

### 推理速度

| 方法 | 吞吐量 (token/s) | 显存 |
|------|------------------|------|
| FP16 | ~30 | 14GB |
| GPTQ-4bit | ~45 | 6GB |
| AWQ-4bit | **~55** | **5.5GB** |

AWQ 推理更快，因为无需复杂的 Hessian 计算。

## 量化速度对比

| 模型 | GPTQ | AWQ |
|------|------|-----|
| 7B | ~15分钟 | **~5分钟** |
| 13B | ~30分钟 | **~10分钟** |
| 70B | ~2小时 | **~40分钟** |

AWQ 量化速度约为 GPTQ 的 2-3 倍。

## 与其他量化方案对比

| 特性 | AWQ | [[gptq\|GPTQ]] | [[gguf\|GGUF]] | bitsandbytes |
|------|-----|------|------|-------------|
| 量化速度 | **快** | 慢 | 最快 | 即时 |
| 推理精度 | **最高** | 高 | 好 | 好 |
| GPU 推理 | ✅ | ✅ | 部分 | ✅ |
| CPU 推理 | ❌ | ❌ | ✅ | ❌ |
| 支持训练 | ❌ | ❌ | ❌ | ✅ |
| 实现复杂度 | **低** | 高 | 低 | 低 |

## 优缺点

| 优点 | 缺点 |
|------|------|
| 精度最高 (4-bit) | 不支持 CPU |
| 量化速度快 | 不支持训练 |
| 推理效率高 | 模型兼容性偶有问题 |
| 实现简单 | 需要 GPU 校准 |

## 适用场景

- GPU 推理，追求最佳精度
- 快速量化部署
- vLLM / TensorRT-LLM 部署
- 精度敏感应用

## 相关资源

- AutoAWQ: https://github.com/casper-hansen/AutoAWQ
- AWQ 论文: https://arxiv.org/abs/2306.00978
- 预量化模型: https://huggingface.co/TheBloke (搜索 AWQ)

## 相关内容

- [[gptq|GPTQ]] — 另一种 GPU 量化方案
- [[gguf|GGUF]] — CPU 推理格式
- [[qlora|QLoRA]] — 训练时量化
- [[bitsandbytes]] — 动态量化库
