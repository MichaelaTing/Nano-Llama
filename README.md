# Nano Llama

Nano Llama is a minimal Llama-style inference project built with pure NumPy. It is designed for learning how Transformer blocks, attention, RoPE, and KV cache work in a clean and readable implementation.

This example uses Andrej Karpathy's [stories15M model](https://github.com/karpathy/llama2.c?tab=readme-ov-file#models) to demonstrate lightweight text generation.

## Features

- Pure NumPy implementation with a simple structure
- Includes tokenization, weight loading, and autoregressive generation
- Great for studying Llama architecture and inference basics

## Usage

```shell
python llama3.py "I have a dream"
```

The script will print generated text and report the token count and generation speed.

## Files

- [llama3.py](llama3.py): main model and inference flow
- [config.py](config.py): model configuration
- [tokenizer.py](tokenizer.py): token encoding and decoding
- [utils.py](utils.py): helper functions such as weight loading