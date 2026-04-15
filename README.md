# Llama3 NumPy 极简实现

这是一个基于 NumPy 编写的 Llama 3 极简推理项目，适合用来学习 Transformer、注意力机制、RoPE 以及 KV Cache 的核心实现方式。

当前示例默认加载 Andrej Karpathy 发布的 [stories15M model](https://github.com/karpathy/llama2.c?tab=readme-ov-file#models)，可以直接体验一个轻量级的文本生成流程。

## 项目特点

- 纯 NumPy 实现，结构直观，便于阅读
- 包含 Tokenizer、模型参数加载和文本生成流程
- 适合作为 Llama 结构入门与推理原理学习示例

## 使用方式

```shell
python llama3.py "I have a dream"
```

运行后会输出模型续写结果，并统计生成速度与 Token 数量。

## 文件说明

- [llama3.py](llama3.py)：模型主体与推理流程
- [config.py](config.py)：模型配置
- [tokenizer.py](tokenizer.py)：分词编码与解码
- [utils.py](utils.py)：权重加载等辅助逻辑