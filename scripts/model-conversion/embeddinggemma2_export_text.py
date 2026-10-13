# Adapted from EverPaper/embeddinggemma-2-coreai@305fc2a96790d9343f38038aeca211dfca56d9a5
# Copyright 2026 Everpaper. Licensed under the Apache License, Version 2.0 (see ../../Sources/MereRunCoreMLModel/LICENSE).
#
# Re-authoring of the EmbeddingGemma 2 text encoder architecture as described by
# modeling_embedding_gemma2.py / configuration_embedding_gemma2.py in Hugging Face transformers 5.19
# (Copyright 2026 The HuggingFace Team, Apache-2.0). Model weights: Google DeepMind (Gemma Team), Apache-2.0,
# https://huggingface.co/google/embeddinggemma-2 — see ../NOTICE for the full attribution and modification list.
"""EmbeddingGemma 2 text encoder re-authored for Core AI export (standard [B, S, D] layout, batch 1).

Everything the sentence-transformers pipeline does is inside one graph:
  input_ids [1, S] int32 + attention_mask [1, S] (1 = token, 0 = right padding)
  -> scaled token embedding (sqrt(512) folded into the fp16 table)
  -> projection-only per-layer embeddings (PLE)
  -> 24 bidirectional encoder layers (5 sliding-window-512 : 1 full, GQA 4 heads / 2 kv (sliding, hd 256) or 1 kv (full, hd 512))
  -> final RMSNorm -> 512->768 projection -> masked mean pooling -> L2 normalization
  -> embedding [1, 768]

Precision modes (PRECISION_MODE, set before construction):
  "ane"  every op fp16 (Neural Engine eligible). RMSNorm and the final L2 norm divide the row by its max-abs value
         before squaring (the model's activations reach ~2000, whose square overflows fp16; this rescaling idea comes
         from the README of the FluidInference Core ML port of this model and was re-derived here).
         Softmax in fp16 (scores are bounded because q/k are RMS-normalized with scale 1.0).
  "gpu"  fp16 weights/matmuls, fp32 islands for every RMSNorm, the softmax, the pooling and the L2 norm
         (the HF reference computes the norms and the softmax in fp32).
  "fp32" the whole model in fp32 (used by the expanded mere.run Core ML export).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PRECISION_MODE = "ane"  # "ane" | "gpu" | "fp32"
GQA_MODE = "grouped5d"  # "expand" (HF repeat_kv: expand+reshape) | "grouped5d" (no repeat, 5-D batched matmul) | "repeat_interleave"
# Global (full-attention, MQA: 1 kv head, head_dim 512) layers 5/11/17/23. "grouped5d" = the GQA_MODE path.
# "mqa4d" (the released bundle) = plain 4-D matmul with k/v broadcast over the heads. With grouped5d the Neural
# Engine compiler rejects the MQA matmul ("Unsupported mps.matmul op for this ANE architecture"), every global layer
# becomes a GPU region, and MPSGraph then places the short buckets entirely on the GPU; mqa4d keeps one GPU region
# (the token gather) and everything else on the ANE. Mathematically identical (fp32 max|diff| 3e-7).
GLOBAL_ATTN_MODE = __import__("os").environ.get("EG2_GLOBAL_ATTN", "mqa4d")
# repeat_interleave made the Neural Engine compiler's fused SDPA fail on the MQA layers (1 kv head, 4 q heads):
# "'mps_spi.sdpa' op failed: grouping for the value tensor does not match the one available on the key tensor".
MASK_VALUE = -10000.0  # finite (ANE softmax mishandles -inf); fp16-exact
NORM_FLOOR = 1e-3  # floor for the row max-abs in "ane" RMSNorm (sqrt(eps) = 1e-3 for eps 1e-6)


class Cfg:
    def __init__(self, model_dir: str):
        c = json.loads((Path(model_dir) / "config.json").read_text())["text_config"]
        self.hidden = c["hidden_size"]  # 512
        self.inter = c["intermediate_size"]  # 2048
        self.n_layers = c["num_hidden_layers"]  # 24
        self.n_heads = c["num_attention_heads"]  # 4
        self.layer_types = c["layer_types"]
        self.sliding_window = c["sliding_window"]  # 512 (inclusive distance)
        self.eps = c["rms_norm_eps"]
        self.vocab = c["vocab_size"]
        self.embed_dim = c["embedding_dim"]  # 768
        self.ple_dim = c["hidden_size_per_layer_input"]  # 512
        self.act = c["hidden_activation"]
        assert self.act == "gelu_pytorch_tanh", self.act
        self.rope_theta = {k: v["rope_theta"] for k, v in c["rope_parameters"].items()}
        per = c.get("per_layer_config", {})
        self.head_dim = [per.get(f"{i:02d}", {}).get("head_dim", c["head_dim"]) for i in range(self.n_layers)]
        self.kv_heads = [per.get(f"{i:02d}", {}).get("num_key_value_heads", c["num_key_value_heads"])
                         for i in range(self.n_layers)]


def _f16(x: float) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float16)


def _recip(m: torch.Tensor) -> torch.Tensor:
    """1/m for m > 0 without a divide: the Neural Engine compiler rejects `mps.divide` (falls back to the GPU);
    rsqrt is native. rsqrt(m)*rsqrt(m) stays in fp16 range for m in [6e-5, 65504]."""
    r = torch.rsqrt(m)
    return r * r


class RMSNorm(nn.Module):
    """Gemma RMSNorm: x * rsqrt(mean(x^2) + eps) * w  (w stored as the raw weight, no +1)."""

    def __init__(self, dim: int, eps: float, with_scale: bool = True):
        super().__init__()
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.eps = eps
        self.register_buffer("eps16", _f16(eps), persistent=False)
        self.register_buffer("sqrt_eps16", _f16(math.sqrt(eps)), persistent=False)
        self.register_buffer("floor16", _f16(NORM_FLOOR), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if PRECISION_MODE == "ane":
            inv_m = _recip(torch.maximum(x.abs().amax(-1, keepdim=True), self.floor16))
            u = x * inv_m
            r = self.sqrt_eps16 * inv_m
            inv = torch.rsqrt((u * u).mean(-1, keepdim=True) + r * r)
            y = u * inv
            return y * self.weight if self.with_scale else y
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.with_scale:
            y = y * self.weight.float()
        return y.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Cfg, i: int):
        super().__init__()
        self.hd = cfg.head_dim[i]
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.kv_heads[i]
        self.groups = self.n_heads // self.n_kv
        self.q_proj = nn.Linear(cfg.hidden, cfg.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.hd, cfg.hidden, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.eps)
        self.k_norm = RMSNorm(self.hd, cfg.eps)
        self.v_norm = RMSNorm(self.hd, cfg.eps, with_scale=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv, self.hd).transpose(1, 2)
        q, k, v = self.q_norm(q), self.k_norm(k), self.v_norm(v)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        # Core ML macOS 15 has fused SDPA with the standard 1/sqrt(head_dim)
        # scale. Compensate on q to retain this model's scale of 1.0. Use
        # float32 and repeat GQA keys explicitly; no quadratic score output.
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
        o = F.scaled_dot_product_attention(q * math.sqrt(self.hd), k, v, attn_mask=bias)
        o = o.transpose(1, 2).reshape(B, S, self.n_heads * self.hd)
        return self.o_proj(o)


class MLP(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden, cfg.inter, bias=False)
        self.up_proj = nn.Linear(cfg.hidden, cfg.inter, bias=False)
        self.down_proj = nn.Linear(cfg.inter, cfg.hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class PLEBlock(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.per_layer_input_gate = nn.Linear(cfg.hidden, cfg.ple_dim, bias=False)
        self.per_layer_projection = nn.Linear(cfg.ple_dim, cfg.hidden, bias=False)
        self.post_per_layer_input_norm = RMSNorm(cfg.hidden, cfg.eps)

    def forward(self, x: torch.Tensor, ple: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.per_layer_input_gate(x), approximate="tanh") * ple
        return x + self.post_per_layer_input_norm(self.per_layer_projection(h))


class Layer(nn.Module):
    def __init__(self, cfg: Cfg, i: int):
        super().__init__()
        self.self_attn = Attention(cfg, i)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden, cfg.eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden, cfg.eps)
        self.pre_feedforward_layernorm = RMSNorm(cfg.hidden, cfg.eps)
        self.post_feedforward_layernorm = RMSNorm(cfg.hidden, cfg.eps)
        self.ple_block = PLEBlock(cfg)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x, ple, cos, sin, bias):
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cos, sin, bias))
        x = x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        x = self.ple_block(x, ple)
        return x * self.layer_scalar


class EmbeddingGemma2Text(nn.Module):
    def __init__(self, cfg: Cfg, max_seq: int = 8192):
        super().__init__()
        self.cfg = cfg
        self.max_seq = max_seq
        self.embed_tokens = nn.Embedding(cfg.vocab, cfg.hidden)  # sqrt(hidden) folded in at load
        self.per_layer_model_projection = nn.Linear(cfg.hidden, cfg.n_layers * cfg.ple_dim, bias=False)  # scale folded
        self.per_layer_projection_norm = RMSNorm(cfg.ple_dim, cfg.eps)
        self.layers = nn.ModuleList([Layer(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden, cfg.eps)
        self.embedding_projection = nn.Linear(cfg.hidden, cfg.embed_dim, bias=False)
        # RoPE tables [max_seq, hd] per (layer type); head dims differ: sliding 256 @ theta 1e4, full 512 @ theta 1e6
        for lt in ("sliding_attention", "full_attention"):
            hd = cfg.head_dim[cfg.layer_types.index(lt)]
            inv = 1.0 / (cfg.rope_theta[lt] ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
            fr = torch.arange(max_seq, dtype=torch.float32)[:, None] * inv[None]
            emb = torch.cat((fr, fr), dim=-1)
            self.register_buffer(f"cos_{lt}", emb.cos(), persistent=False)
            self.register_buffer(f"sin_{lt}", emb.sin(), persistent=False)
        self.register_buffer("mask_value", _f16(MASK_VALUE), persistent=False)
        self.register_buffer("one", _f16(1.0), persistent=False)
        self.register_buffer("zero", _f16(0.0), persistent=False)
        self.register_buffer("floor16", _f16(NORM_FLOOR), persistent=False)
        self.register_buffer("window", torch.tensor(cfg.sliding_window, dtype=torch.int32), persistent=False)

    def _biases(self, attention_mask: torch.Tensor, S: int):
        pad = (self.one - attention_mask) * self.mask_value  # [1, S]  0 for tokens, -1e4 for padding keys
        full = pad[:, None, None, :]  # [1, 1, 1, S] broadcast over heads and queries
        pos = torch.arange(S, device=attention_mask.device, dtype=torch.int32)
        dist = (pos[:, None] - pos[None, :]).abs()
        band = torch.where(dist <= self.window, self.zero, self.mask_value)  # [S, S]
        sliding = full + band[None, None]
        return full, sliding

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        S = input_ids.shape[1]
        x = self.embed_tokens(input_ids)  # [1, S, 512], already scaled
        ple = self.per_layer_projection_norm(
            self.per_layer_model_projection(x).view(1, S, self.cfg.n_layers, self.cfg.ple_dim))
        full, sliding = self._biases(attention_mask, S)
        cs = {}
        for lt in ("sliding_attention", "full_attention"):
            cs[lt] = (getattr(self, f"cos_{lt}")[:S].to(x.dtype)[None, None], getattr(self, f"sin_{lt}")[:S].to(x.dtype)[None, None])
        for i, layer in enumerate(self.layers):
            lt = self.cfg.layer_types[i]
            x = layer(x, ple[:, :, i, :], cs[lt][0], cs[lt][1], sliding if lt == "sliding_attention" else full)
        x = self.embedding_projection(self.norm(x))  # [1, S, 768]
        # masked mean pooling: weights sum to 1, so no large partial sums in fp16
        if PRECISION_MODE == "ane":
            w = attention_mask * _recip(attention_mask.sum(-1, keepdim=True))  # [1, S] (count exact in fp16 up to 2048)
            pooled = torch.matmul(w[:, None, :], x)[:, 0, :]  # [1, 768]
            u = pooled * _recip(torch.maximum(pooled.abs().amax(-1, keepdim=True), self.floor16))
            return u * torch.rsqrt((u * u).sum(-1, keepdim=True))
        xf = x.float()
        wf = attention_mask.float()
        pooled = (xf * wf[:, :, None]).sum(1) / wf.sum(-1, keepdim=True)
        return (pooled / pooled.norm(dim=-1, keepdim=True)).to(x.dtype)


def load_model(model_dir: str, dtype: torch.dtype, max_seq: int = 8192) -> EmbeddingGemma2Text:
    from safetensors import safe_open

    cfg = Cfg(model_dir)
    m = EmbeddingGemma2Text(cfg, max_seq)
    sd = {}
    with safe_open(str(Path(model_dir) / "model.safetensors"), "pt") as f:
        for k in f.keys():
            if not k.startswith("language_model."):
                continue
            t = f.get_tensor(k).float()
            k = k[len("language_model."):]
            if k == "embed_tokens.weight":
                t = t * math.sqrt(cfg.hidden)
            elif k == "ple.per_layer_model_projection.weight":
                k = "per_layer_model_projection.weight"
                t = t * cfg.hidden ** -0.5
            elif k == "ple.per_layer_projection_norm.weight":
                k = "per_layer_projection_norm.weight"
            sd[k] = t.to(dtype)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not any(s in k for s in ("cos_", "sin_", "eps16", "sqrt_eps16", "floor16", "mask_value", "one", "zero", "window"))]
    assert not missing and not unexpected, (missing, unexpected)
    return m.to(dtype).eval()
