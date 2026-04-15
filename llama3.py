from __future__ import annotations

"""Llama 3 的纯 NumPy 推理实现。

这份代码参考了纯 NumPy 拆解 Llama 3 的讲解思路，重点把下面几部分串起来：
1. 先预计算 RoPE 所需的 cos / sin
2. 在每个 Transformer Block 中执行 RMSNorm、Attention、FeedForward
3. 推理时利用 KV Cache，把生成过程拆成 Prefill 和 Decode 两个阶段

注释中的形状缩写含义：
- B: batch size
- L: 当前序列长度
- D: hidden size
- HN: attention head 数量
- KVHN: key/value head 数量
- HD: 每个 head 的维度
- VS: 词表大小
- M: 最大序列长度
"""

import math
import sys
import time
from typing import TypeVar, Generic, Optional

import numpy as np

from config import ModelArgs
from tokenizer import Tokenizer
from utils import load_parameters

Shape = TypeVar("Shape")


class Array(np.ndarray, Generic[Shape]): ...


# 对最后一维做 softmax，把分数归一化成概率分布。
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# SiLU 也叫 Swish，是 Llama 前馈网络中使用的激活函数。
def silu(x):
    return x * (1 / (1 + np.exp(-x)))


# 为 Query 和 Key 应用旋转位置编码（RoPE）。
# RoPE 不是直接加到 embedding 上，而是在 Q/K 投影后再做旋转。
def apply_rotary_emb(
    xq: Array["B, L or 1, QHN,  HD"],
    xk: Array["B, L or 1, KVHN, HD"],
    freqs_cos: Array["L or 1, HD//2"],
    freqs_sin: Array["L or 1, HD//2"]
):
    # 把最后一维按 2 个一组拆开，便于把实部和虚部配对处理。
    xqri: Array["B, L or 1, QHN,  HD//2, 2"] = xq.reshape(xq.shape[:-1] + (-1, 2))
    xkri: Array["B, L or 1, KVHN, HD//2, 2"] = xk.reshape(xk.shape[:-1] + (-1, 2))

    # 拆成实部和虚部，后面按旋转矩阵公式计算。
    xq_r, xq_i = np.split(xqri, 2, axis=-1)
    xq_r: Array["B, L or 1, QHN,  HD//2"] = xq_r.squeeze(-1)
    xq_i: Array["B, L or 1, QHN,  HD//2"] = xq_i.squeeze(-1)

    xk_r, xk_i = np.split(xkri, 2, axis=-1)
    xk_r: Array["B, L or 1, KVHN, HD//2"] = xk_r.squeeze(-1)
    xk_i: Array["B, L or 1, KVHN, HD//2"] = xk_i.squeeze(-1)

    # 扩展 cos / sin 维度，让它们能与多头张量自动广播。
    freqs_cos: Array["B, L or 1, 1, HD//2"] = np.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin: Array["B, L or 1, 1, HD//2"] = np.expand_dims(freqs_sin, axis=(0, 2))

    # 按旋转公式应用到 Q 和 K 上。
    xq_out_r: Array["B, L or 1, QHN,  HD//2"] = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i: Array["B, L or 1, QHN,  HD//2"] = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r: Array["B, L or 1, KVHN, HD//2"] = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i: Array["B, L or 1, KVHN, HD//2"] = xk_r * freqs_sin + xk_i * freqs_cos

    # 把旋转后的结果拼回去，恢复原始 head_dim 形状。
    xq_out: Array["B, L or 1, QHN,  HD//2, 2"] = np.stack([xq_out_r, xq_out_i], axis=-1)
    xk_out: Array["B, L or 1, KVHN, HD//2, 2"] = np.stack([xk_out_r, xk_out_i], axis=-1)
    xq_out: Array["B, L or 1, QHN,  HD"] = xq_out.reshape(xq_out.shape[:-2] + (-1,))
    xk_out: Array["B, L or 1, KVHN, HD"] = xk_out.reshape(xk_out.shape[:-2] + (-1,))

    return xq_out, xk_out


# GQA 中 Q 头数可能多于 KV 头数，因此需要把 K/V 按倍数复制。
# 这个 stories15M 模型本身没有真正使用 GQA，所以多数情况下 n_rep == 1。
def repeat_kv(x: Array["B, L, KVHN, HD"], n_rep: int):
    if n_rep == 1:
        return x
    z: Array["B, L, QHN, HD"] = np.repeat(x, n_rep, axis=2)
    return z


# Llama 的前馈层采用 SwiGLU：
# 一路做 gate（经过 SiLU），一路做 up projection，二者逐元素相乘后再降维。
class FeedForward:
    def __init__(self, up_weight: Array["FD, D"], gate_weight: Array["FD, D"], down_weight: Array["D, FD"]):
        self.up_weight = up_weight.T
        self.gate_weight = gate_weight.T
        self.down_weight = down_weight.T

    def __call__(self, x: Array["B, L or 1, D"]):
        # FD 一般会比 D 更大，用来提升中间层表达能力。
        swish: Array["B, L or 1, FD"] = silu(x @ self.gate_weight)
        x_V: Array["B, L or 1, FD"] = x @ self.up_weight
        x: Array["B, L or 1, FD"] = swish * x_V
        x: Array["B, L or 1, D"] = x @ self.down_weight
        return x


# RMSNorm 使用均方根来归一化激活值。
# 与 BatchNorm / LayerNorm 相比，它更轻量，也很适合 LLM。
class RMSNorm:
    def __init__(self, weight: Array["H"], eps: float):
        self.weight = weight
        self.eps = eps

    def __call__(self, x: Array["B, L or 1, D"]):
        # 先计算均方，再开根号进行缩放，最后乘上可学习参数。
        z: Array["B, L or 1, 1"] = (x ** 2).mean(-1, keepdims=True) + self.eps
        z: Array["B, L or 1, D"] = x / np.sqrt(z)
        return z * self.weight


# 自注意力模块。
# 这里包含 QKV 投影、RoPE、KV Cache、GQA 以及最终的输出投影。
class Attention:
    def __init__(
        self,
        q_weight: Array["D, D"],
        k_weight: Array["D, D"],
        v_weight: Array["D, D"],
        o_weight: Array["D, D"],
        args: ModelArgs
    ):
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.q_weight = q_weight.T
        self.k_weight = k_weight.T
        self.v_weight = v_weight.T
        self.o_weight = o_weight.T

        # 提前分配 KV Cache。
        # 生成时，历史 token 的 K/V 不必重复计算，只需要把新结果写入缓存。
        self.cache_k = np.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim))
        self.cache_v = np.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim))

    def __call__(
        self,
        x: Array["B, L or 1, D"],
        start_pos: int,
        mask: Optional[Array["L, L"]],
        freqs_cos: Array["L or 1, HD//2"],
        freqs_sin: Array["L or 1, HD//2"]
    ):
        B, L, _ = x.shape

        # Llama 与一些 GPT 实现不同，Q、K、V 各自使用独立权重。
        xq: Array["B, L or 1, D"] = x @ self.q_weight
        xk: Array["B, L or 1, D"] = x @ self.k_weight
        xv: Array["B, L or 1, D"] = x @ self.v_weight

        # 按多头形式 reshape，拆成 [head 数, 每头维度]。
        xq: Array["B, L or 1, QHN,  HD"] = xq.reshape(B, L, self.n_local_heads, self.head_dim)
        xk: Array["B, L or 1, KVHN, HD"] = xk.reshape(B, L, self.n_local_kv_heads, self.head_dim)
        xv: Array["B, L or 1, KVHN, HD"] = xv.reshape(B, L, self.n_local_kv_heads, self.head_dim)

        # 对 Q/K 应用位置编码，使注意力具备相对位置信息。
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 将本轮新得到的 K/V 写入缓存，再取出截至当前长度的全部历史信息。
        self.cache_k[:B, start_pos: start_pos + L] = xk
        self.cache_v[:B, start_pos: start_pos + L] = xv
        ks: Array["B, L, KVHN, HD"] = self.cache_k[:B, : start_pos + L]
        vs: Array["B, L, KVHN, HD"] = self.cache_v[:B, : start_pos + L]

        # 如果启用了 GQA，就把较少的 KV 头复制到与 Q 头一致。
        xk: Array["B, L, HN, HD"] = repeat_kv(ks, self.n_rep)
        xv: Array["B, L, HN, HD"] = repeat_kv(vs, self.n_rep)

        # 转成 attention 计算习惯的维度顺序：[B, HN, L, HD]。
        xq: Array["B, HN, L or 1, HD"] = xq.transpose(0, 2, 1, 3)
        xk: Array["B, HN, L, HD"] = xk.transpose(0, 2, 1, 3)
        xv: Array["B, HN, L, HD"] = xv.transpose(0, 2, 1, 3)

        # Scaled Dot-Product Attention: softmax(QK^T / sqrt(d_k))V
        attention: Array["B, HN, L or 1, L"] = xq @ xk.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)

        # 只有在 prefill 阶段需要显式 causal mask，避免看到未来 token。
        if mask is not None:
            attention = attention + mask[None, None, :, :]

        attention = softmax(attention)
        output: Array["B, HN, L or 1, HD"] = attention @ xv

        # 合并多头后，再做一次输出投影。
        output: Array["B, L or 1, D"] = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output: Array["B, L or 1, D"] = output @ self.o_weight

        return output


# 一个标准的 Transformer Block：
# Pre-Norm -> Attention -> 残差 -> Pre-Norm -> FFN -> 残差
class TransformerBlock:
    def __init__(self, weight: dict, layer_id: int, args: ModelArgs):
        self.attention = Attention(
            weight.get(f"model.layers.{layer_id}.self_attn.q_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.k_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.v_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.o_proj.weight"),
            args
        )
        self.feed_forward = FeedForward(
            weight.get(f"model.layers.{layer_id}.mlp.up_proj.weight"),
            weight.get(f"model.layers.{layer_id}.mlp.gate_proj.weight"),
            weight.get(f"model.layers.{layer_id}.mlp.down_proj.weight"),
        )
        self.input_layernorm = RMSNorm(
            weight.get(f"model.layers.{layer_id}.input_layernorm.weight"),
            eps=args.norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weight.get(f"model.layers.{layer_id}.post_attention_layernorm.weight"),
            eps=args.norm_eps
        )

    def __call__(
        self,
        x: Array["B, L or 1, D"],
        start_pos: int,
        mask: Array["L, L"],
        freqs_cos: Array["L or 1, HD//2"],
        freqs_sin: Array["L or 1, HD//2"]
    ):
        # 先做输入归一化，再送入注意力层。
        norm_x: Array["B, L or 1, D"] = self.input_layernorm(x)
        h1: Array["B, L or 1, D"] = self.attention(norm_x, start_pos, mask, freqs_cos, freqs_sin)
        z = x + h1

        # 第二次归一化后进入 SwiGLU 前馈层。
        norm_z = self.post_attention_layernorm(z)
        h2: Array["B, L or 1, D"] = self.feed_forward(norm_z)
        out = z + h2

        return out


# 模型主体：加载 embedding、各层 block、最终 norm 和词表投影层。
class Llama:
    def __init__(self, model_path: str, args: ModelArgs):
        self.args = args

        weight = load_parameters(model_path)
        self.tok_embedding: Array["VS, D"] = weight.get("model.embed_tokens.weight")

        # 预计算整段最大长度的 RoPE 频率。
        # 这些值在一次请求里可复用，因此适合提前算好并缓存下来。
        base = 10000
        head_dim = args.dim // args.n_heads
        inv_freq: Array["HD//2"] = 1.0 / (base ** (np.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim))
        t: Array["M"] = np.arange(args.max_seq_len)
        freqs: Array["M, HD//2"] = np.outer(t, inv_freq)
        self.freqs_cos: Array["M, HD//2"] = np.cos(freqs)
        self.freqs_sin: Array["M, HD//2"] = np.sin(freqs)

        self.layers = []
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(weight, layer_id, args))

        self.norm = RMSNorm(weight.get("model.norm.weight"), eps=args.norm_eps)
        self.lm_head_weight: Array["D, VS"] = weight.get("lm_head.weight").T

        del weight

    def __call__(self, input_ids: Array["B, L"], start_pos: int):
        _, L = input_ids.shape

        # 把 token id 查表映射到词向量。
        h: Array["B, L or 1, D"] = self.tok_embedding[input_ids]

        # 取出当前位置对应的 cos / sin 片段，供本轮 Q/K 使用。
        freqs_cos: Array["L or 1, HD//2"] = self.freqs_cos[start_pos: start_pos + L]
        freqs_sin: Array["L or 1, HD//2"] = self.freqs_sin[start_pos: start_pos + L]

        # causal mask 只在首次整段输入时构建一次；
        # 后续 decode 每次只有一个 token，因此无需重复构造完整 mask。
        mask: Array["L, L"] = None
        if L > 1:
            mask = np.full((L, L), float("-inf"))
            mask = np.triu(mask, k=1)
            mask = np.concatenate([np.zeros((L, start_pos)), mask], axis=1)

        # 依次通过所有 Transformer 层。
        for i, layer in enumerate(self.layers):
            h: Array["B, L or 1, D"] = layer(h, start_pos, mask, freqs_cos, freqs_sin)

        # 输出端再做一次 RMSNorm。
        h: Array["B, L or 1, D"] = self.norm(h)

        # 推理时只关心最后一个位置的 logits，这样速度更快。
        logit: Array["B, 1, VS"] = h[:, [-1], :] @ self.lm_head_weight
        return logit

    def generate(self, input_ids: Array["B, L"], max_new_tokens: int):
        # 自回归生成分成两段：
        # 1. Prefill：把整个 prompt 一次性送入模型，建立 KV Cache
        # 2. Decode：之后每次只输入一个新 token，不断向后生成
        _, L = input_ids.shape
        for i, curr_pos in enumerate(range(L, max_new_tokens)):
            if i == 0:
                inputs = input_ids
                pos = 0
            else:
                inputs = next_id
                pos = curr_pos

            logits: Array["B, 1, VS"] = self(inputs, pos)

            # 这里使用 greedy decoding，直接取概率最大的 token。
            # 若需要更自然的结果，可以在这里扩展 top-k / top-p sampling。
            next_id = logits[:, -1, :].argmax(-1, keepdims=True)
            yield next_id


if __name__ == '__main__':
    # 初始化模型配置、分词器和权重文件。
    args = ModelArgs()

    tokenizer = Tokenizer("./tokenizer.model.np")
    model = Llama("./stories15M.model.npz", args)

    # 若命令行没有传入 prompt，就用一个默认例子演示。
    if len(sys.argv) == 1:
        prompt = "I have a dream"
    else:
        prompt = sys.argv[1]

    print(f"\n{prompt}", end="")
    input_ids = np.array([tokenizer.encode(prompt)])
    start = time.time()
    _, L = input_ids.shape

    # 逐 token 生成，并在遇到特殊结束符时停止。
    for id in model.generate(input_ids, args.max_new_tokens):
        L += 1
        output_id = id[0].tolist()
        if output_id[-1] in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        print(tokenizer.decode(output_id), end="")
        sys.stdout.flush()

    elapsed = time.time() - start
    print(f"\n\nToken count: {L}, elapsed: {elapsed:.2f}s, {round(L / elapsed)} tokens/s")
