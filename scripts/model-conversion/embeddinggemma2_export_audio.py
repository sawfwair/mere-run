# Adapted from EverPaper/embeddinggemma-2-coreai@305fc2a96790d9343f38038aeca211dfca56d9a5
# Copyright 2026 Everpaper. Licensed under the Apache License, Version 2.0 (see ../../Sources/MereRunCoreMLModel/LICENSE).
#
# Re-authoring of the Gemma 4 audio encoder (Gemma4AudioModel: a USM-style conformer) and the EmbeddingGemma 2
# audio embedder as described by modeling_gemma4.py / modeling_embedding_gemma2.py in Hugging Face transformers
# 5.19 (Copyright 2026 The HuggingFace Team, Apache-2.0). Model weights: Google DeepMind (Gemma Team),
# Apache-2.0, https://huggingface.co/google/embeddinggemma-2 — see ../NOTICE for the attribution and
# modification list.
"""EmbeddingGemma 2 audio encoder re-authored for Core AI export: log-mel frames -> soft tokens in the text space.

Inputs (one function per frame bucket Tb, functions `audio_<Tb>`):
  features    [1, Tb, 128] fp16   log-mel frames of the upstream feature extractor (host side: python/eg2_audio_frontend.py,
                                  swift AudioFrontend), the first n_frames rows real, the rest zero
  frame_mask  [1, Tb]      fp16   1 for the first n_frames rows, 0 after
  half_mask   [1, Tb/2]    fp16   1 for the first ceil(n_frames/2) rows (mask after the first stride-2 conv)
  token_mask  [1, Tb/4]    fp16   1 for the first ceil(ceil(n_frames/2)/2) rows (= the soft-token count)
Output:
  soft_tokens [1, Tb/4, 512] fp16; rows at index >= n_tokens are padding and must be ignored.

The graph: features * frame_mask -> Conv2d 1->128 (3x3, stride 2) + LayerNorm over channels + ReLU -> * half_mask
-> Conv2d 128->32 (3x3, stride 2) + LayerNorm + ReLU -> [T, F=32, C=32] flattened to 1024 -> Linear 1024->1024
-> 12 conformer layers (half-step FFN, local causal attention with relative position bias, light depthwise
convolution, half-step FFN, RMSNorm) -> Linear 1024->1536 (+bias) -> RMSNorm without scale -> Linear 1536->512.

Differences from the upstream graph (all mathematically identical in exact arithmetic):
  - the chunked local attention (pad / unfold / rel_shift view trick / 5-D blocks) is written in its banded form:
    every query attends to itself and the 11 previous tokens (the mask of the upstream chunked attention lets each
    position see exactly these 12 keys), logit[t, d] = q_t . (k_{t-d} + relk_d) for d = 0..11, where
    relk_d = relative_k_proj(sinusoid(d)) is a per-layer constant; keys and values at offset d are the tensor
    shifted by d with pad + slice. No gather, no rank-5 tensors.
  - masked_fill(-1e9) on padded keys becomes an additive bias (-10000 in fp16, finite) added after the softcap;
    rows whose 12 keys are all padding (only possible beyond the valid tokens) stay finite.
  - the q scale (128^-0.5 / ln 2 * softplus(per_dim_scale), per head dimension) and the k scale (ln(1+e) / ln 2)
    are folded into q_proj / k_proj and their output clip bounds: clamp(Wx, lo, hi) * s == clamp(sWx, s*lo, s*hi)
    for s > 0. The clip bounds of all 120 clipped linears stay as min/max ops.
  - the gradient-clipping clamps (+-1e10) are no-ops in inference and are dropped.
  - the SubSampleConvProjection LayerNorms (over the channel axis of the NCHW tensor) are written as mean/variance
    over dim 1; GLU as slice * sigmoid; the causal depthwise Conv1d as a grouped conv with 4 samples of left pad.
  - fp16 "ane" precision mode as in eg2_text.py: every RMSNorm divides the row by its max-abs value before squaring
    (the residual stream reaches ~412, whose square overflows fp16), every division is rsqrt(x)*rsqrt(x).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PRECISION_MODE = "ane"  # "ane" (all fp16, Neural Engine) | "fp32" (correctness gate of the re-authoring)
MASK_VALUE = -10000.0
NORM_FLOOR = 1e-3
CTX = 12  # keys per query: self + 11 past
N_MELS = 128
FRAME_BUCKETS = [192, 384, 1024, 1536, 3000]  # frames (10 ms); tokens = frames / 4; 3000 frames = 30 s (2999 valid frames)


class Cfg:
    def __init__(self, model_dir: str):
        c = json.loads((Path(model_dir) / "config.json").read_text())["audio_config"]
        t = json.loads((Path(model_dir) / "config.json").read_text())["text_config"]
        self.hidden = c["hidden_size"]  # 1024
        self.n_layers = c["num_hidden_layers"]  # 12
        self.n_heads = c["num_attention_heads"]  # 8
        self.head_dim = self.hidden // self.n_heads  # 128
        self.eps = c["rms_norm_eps"]  # 1e-6
        self.conv_channels = c["subsampling_conv_channels"]  # [128, 32]
        self.conv_kernel = c["conv_kernel_size"]  # 5
        self.softcap = c["attention_logit_cap"]  # 50
        self.residual_weight = c["residual_weight"]  # 0.5
        self.out_dims = c["output_proj_dims"]  # 1536
        self.text_hidden = t["hidden_size"]  # 512
        assert c["attention_chunk_size"] + c["attention_context_left"] - 1 + c["attention_context_right"] == 2 * CTX
        assert c["attention_context_right"] == 0 and c["hidden_act"] == "silu" and c["use_clipped_linears"]


def _f16(x: float) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float16)


def _recip(m: torch.Tensor) -> torch.Tensor:
    r = torch.rsqrt(m)
    return r * r


class RMSNorm(nn.Module):
    """Gemma RMSNorm (weight used as-is), with the max-abs rescaling in "ane" mode."""

    def __init__(self, dim: int, eps: float, with_scale: bool = True):
        super().__init__()
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.eps = eps
        self.register_buffer("sqrt_eps16", _f16(math.sqrt(eps)), persistent=False)
        self.register_buffer("floor16", _f16(NORM_FLOOR), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if PRECISION_MODE == "ane":
            inv_m = _recip(torch.maximum(x.abs().amax(-1, keepdim=True), self.floor16))
            u = x * inv_m
            r = self.sqrt_eps16 * inv_m
            y = u * torch.rsqrt((u * u).mean(-1, keepdim=True) + r * r)
            return y * self.weight if self.with_scale else y
        xf = x.float()
        y = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            y = y * self.weight.float()
        return y.to(x.dtype)


class ClippedLinear(nn.Module):
    """Linear whose input and output are clamped to bounds stored in the checkpoint (Gemma4ClippableLinear).
    The output bounds are per output feature so that a per-feature scale can be folded in (see load_model)."""

    def __init__(self, din: int, dout: int):
        super().__init__()
        self.linear = nn.Linear(din, dout, bias=False)
        self.register_buffer("input_min", torch.zeros(()))
        self.register_buffer("input_max", torch.zeros(()))
        self.register_buffer("output_min", torch.zeros(dout))
        self.register_buffer("output_max", torch.zeros(dout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.minimum(torch.maximum(x, self.input_min), self.input_max)
        y = self.linear(x)
        return torch.minimum(torch.maximum(y, self.output_min), self.output_max)


class FeedForward(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.pre_layer_norm = RMSNorm(cfg.hidden, cfg.eps)
        self.ffw_layer_1 = ClippedLinear(cfg.hidden, 4 * cfg.hidden)
        self.ffw_layer_2 = ClippedLinear(4 * cfg.hidden, cfg.hidden)
        self.post_layer_norm = RMSNorm(cfg.hidden, cfg.eps)
        self.register_buffer("res_w16", _f16(cfg.residual_weight), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ffw_layer_2(F.silu(self.ffw_layer_1(self.pre_layer_norm(x))))
        return self.post_layer_norm(h) * self.res_w16 + x


class Attention(nn.Module):
    """Local causal attention in banded form: 12 keys per query (self + 11 past) with the relative-position
    constant relk_d added to the key at offset d. q/k scales are folded into the projections."""

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.H, self.D = cfg.n_heads, cfg.head_dim
        self.q_proj = ClippedLinear(cfg.hidden, cfg.hidden)
        self.k_proj = ClippedLinear(cfg.hidden, cfg.hidden)
        self.v_proj = ClippedLinear(cfg.hidden, cfg.hidden)
        self.post = ClippedLinear(cfg.hidden, cfg.hidden)
        self.register_buffer("relk", torch.zeros(CTX, cfg.n_heads, cfg.head_dim))  # relative_k_proj(sinusoid(d))
        self.register_buffer("inv_softcap16", _f16(1.0 / cfg.softcap), persistent=False)
        self.register_buffer("softcap16", _f16(cfg.softcap), persistent=False)
        self.register_buffer("mask_value", _f16(MASK_VALUE), persistent=False)
        self.register_buffer("one", _f16(1.0), persistent=False)
        self.softcap = cfg.softcap

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.H, self.D)
        k = self.k_proj(x).view(B, T, self.H, self.D)
        v = self.v_proj(x).view(B, T, self.H, self.D)
        logits, vs, valid = [], [], []
        for d in range(CTX):
            kd = k if d == 0 else F.pad(k, (0, 0, 0, 0, d, 0))[:, :T]
            vd = v if d == 0 else F.pad(v, (0, 0, 0, 0, d, 0))[:, :T]
            md = token_mask if d == 0 else F.pad(token_mask, (d, 0))[:, :T]
            logits.append((q * (kd + self.relk[d])).sum(-1))  # [B, T, H]
            vs.append(vd)
            valid.append(md)
        lg = torch.stack(logits, -1)  # [B, T, H, CTX]
        if PRECISION_MODE == "ane":
            lg = torch.tanh(lg * self.inv_softcap16) * self.softcap16
            bias = ((self.one - torch.stack(valid, -1)) * self.mask_value)[:, :, None, :]  # [B, T, 1, CTX]
            w = torch.softmax(lg + bias, -1)
        else:
            lg = torch.tanh(lg.float() / self.softcap) * self.softcap
            bias = (1.0 - torch.stack(valid, -1).float())[:, :, None, :] * -1e9
            w = torch.softmax(lg + bias, -1).to(x.dtype)
        o = w[..., 0:1] * vs[0]
        for d in range(1, CTX):
            o = o + w[..., d:d + 1] * vs[d]
        return self.post(o.reshape(B, T, self.H * self.D))


class LightConv(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.pre_layer_norm = RMSNorm(cfg.hidden, cfg.eps)
        self.linear_start = ClippedLinear(cfg.hidden, 2 * cfg.hidden)
        self.depthwise_conv1d = nn.Conv1d(cfg.hidden, cfg.hidden, cfg.conv_kernel, groups=cfg.hidden, bias=False)
        self.conv_norm = RMSNorm(cfg.hidden, cfg.eps)
        self.linear_end = ClippedLinear(cfg.hidden, cfg.hidden)
        self.hidden = cfg.hidden
        self.left_pad = cfg.conv_kernel - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear_start(self.pre_layer_norm(x))
        h = h[..., : self.hidden] * torch.sigmoid(h[..., self.hidden:])
        h = self.depthwise_conv1d(F.pad(h.transpose(1, 2), (self.left_pad, 0))).transpose(1, 2)
        return self.linear_end(F.silu(self.conv_norm(h))) + x


class Layer(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.feed_forward1 = FeedForward(cfg)
        self.norm_pre_attn = RMSNorm(cfg.hidden, cfg.eps)
        self.self_attn = Attention(cfg)
        self.norm_post_attn = RMSNorm(cfg.hidden, cfg.eps)
        self.lconv1d = LightConv(cfg)
        self.feed_forward2 = FeedForward(cfg)
        self.norm_out = RMSNorm(cfg.hidden, cfg.eps)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        x = self.feed_forward1(x)
        x = self.norm_post_attn(self.self_attn(self.norm_pre_attn(x), token_mask)) + x
        x = self.lconv1d(x)
        x = self.feed_forward2(x)
        return self.norm_out(x)


class SubsampleLayer(nn.Module):
    """Conv2d 3x3 stride 2 (no bias) + LayerNorm over the channel axis (no bias) + ReLU, on NCHW [B, C, T, F]."""

    def __init__(self, cin: int, cout: int, eps: float):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False)
        self.norm_weight = nn.Parameter(torch.ones(cout), requires_grad=False)
        self.eps = eps
        self.register_buffer("eps16", _f16(eps), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if PRECISION_MODE == "ane":
            mu = y.mean(1, keepdim=True)
            c = y - mu
            var = (c * c).mean(1, keepdim=True)
            y = c * torch.rsqrt(var + self.eps16) * self.norm_weight[None, :, None, None]
        else:
            yf = y.float()
            mu = yf.mean(1, keepdim=True)
            var = (yf - mu).pow(2).mean(1, keepdim=True)
            y = ((yf - mu) * torch.rsqrt(var + self.eps) * self.norm_weight.float()[None, :, None, None]).to(y.dtype)
        return F.relu(y)


class EmbeddingGemma2Audio(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        c0, c1 = cfg.conv_channels
        self.layer0 = SubsampleLayer(1, c0, cfg.eps)
        self.layer1 = SubsampleLayer(c0, c1, cfg.eps)
        self.input_proj_linear = nn.Linear((N_MELS // 4) * c1, cfg.hidden, bias=False)
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(cfg.n_layers)])
        self.output_proj = nn.Linear(cfg.hidden, cfg.out_dims, bias=True)
        self.pre_projection_norm = RMSNorm(cfg.out_dims, cfg.eps, with_scale=False)
        self.embedding_projection = nn.Linear(cfg.out_dims, cfg.text_hidden, bias=False)

    def forward(self, features, frame_mask, half_mask, token_mask):
        B, T, Fm = features.shape
        x = (features * frame_mask[:, :, None])[:, None]  # [B, 1, T, 128]
        x = self.layer0(x) * half_mask[:, None, :, None]  # [B, 128, T/2, 64]
        x = self.layer1(x)  # [B, 32, T/4, 32]
        C, T4, Fq = x.shape[1], x.shape[2], x.shape[3]
        x = self.input_proj_linear(x.permute(0, 2, 3, 1).reshape(B, T4, Fq * C))  # index f*32 + c
        for layer in self.layers:
            x = layer(x, token_mask)
        x = self.output_proj(x)
        return self.embedding_projection(self.pre_projection_norm(x))  # [B, T/4, 512]


def masks_for(n_frames: int, bucket: int, dtype=torch.float16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(frame_mask [1,Tb], half_mask [1,Tb/2], token_mask [1,Tb/4]) for n_frames valid frames in a bucket."""
    fm = torch.zeros(1, bucket, dtype=dtype)
    hm = torch.zeros(1, bucket // 2, dtype=dtype)
    tm = torch.zeros(1, bucket // 4, dtype=dtype)
    fm[0, :n_frames] = 1
    hm[0, : (n_frames + 1) // 2] = 1
    tm[0, : ((n_frames + 1) // 2 + 1) // 2] = 1
    return fm, hm, tm


def n_tokens(n_frames: int) -> int:
    return ((n_frames + 1) // 2 + 1) // 2


def bucket_for(n_frames: int, buckets=FRAME_BUCKETS) -> int:
    for b in buckets:
        if b >= n_frames:
            return b
    raise ValueError(f"{n_frames} frames exceed the largest bucket {buckets[-1]}")


def relative_keys(cfg: Cfg, rel_w: torch.Tensor) -> torch.Tensor:
    """relk[d] = relative_k_proj(pos_d), pos_d = [sin(d*inv), cos(d*inv)], inv_i = exp(-i ln(1e4)/(512-1)) (fp32)."""
    n = cfg.hidden // 2
    inv = torch.exp(torch.arange(n, dtype=torch.float32) * -(math.log(10000.0) / max(n - 1, 1)))
    d = torch.arange(CTX, dtype=torch.float32)[:, None]
    scaled = d * inv[None, :]
    pos = torch.cat([torch.sin(scaled), torch.cos(scaled)], -1)  # [CTX, 1024]
    return (pos @ rel_w.float().T).view(CTX, cfg.n_heads, cfg.head_dim)


def load_model(model_dir: str, dtype: torch.dtype) -> EmbeddingGemma2Audio:
    from safetensors import safe_open

    cfg = Cfg(model_dir)
    m = EmbeddingGemma2Audio(cfg)
    sd: dict[str, torch.Tensor] = {}
    with safe_open(str(Path(model_dir) / "model.safetensors"), "pt") as f:
        raw = {k: f.get_tensor(k).float() for k in f.keys() if k.startswith("audio_tower.") or k.startswith("embed_audio.")}
    q_scale = (cfg.head_dim ** -0.5) / math.log(2)
    k_scale = math.log(1 + math.e) / math.log(2)

    def clipped(dst: str, src: str, scale: torch.Tensor | float | None = None):
        w = raw[f"{src}.linear.weight"]
        omin = raw[f"{src}.output_min"].expand(w.shape[0]).clone()
        omax = raw[f"{src}.output_max"].expand(w.shape[0]).clone()
        if scale is not None:
            s = torch.as_tensor(scale, dtype=torch.float32)
            assert bool((s > 0).all())
            w = w * s[:, None] if s.ndim else w * s
            omin, omax = omin * s, omax * s
        sd[f"{dst}.linear.weight"] = w
        sd[f"{dst}.input_min"] = raw[f"{src}.input_min"]
        sd[f"{dst}.input_max"] = raw[f"{src}.input_max"]
        sd[f"{dst}.output_min"] = omin
        sd[f"{dst}.output_max"] = omax

    sd["layer0.conv.weight"] = raw["audio_tower.subsample_conv_projection.layer0.conv.weight"]
    sd["layer0.norm_weight"] = raw["audio_tower.subsample_conv_projection.layer0.norm.weight"]
    sd["layer1.conv.weight"] = raw["audio_tower.subsample_conv_projection.layer1.conv.weight"]
    sd["layer1.norm_weight"] = raw["audio_tower.subsample_conv_projection.layer1.norm.weight"]
    sd["input_proj_linear.weight"] = raw["audio_tower.subsample_conv_projection.input_proj_linear.weight"]
    for i in range(cfg.n_layers):
        p = f"audio_tower.layers.{i}"
        for ff in ("feed_forward1", "feed_forward2"):
            clipped(f"layers.{i}.{ff}.ffw_layer_1", f"{p}.{ff}.ffw_layer_1")
            clipped(f"layers.{i}.{ff}.ffw_layer_2", f"{p}.{ff}.ffw_layer_2")
            sd[f"layers.{i}.{ff}.pre_layer_norm.weight"] = raw[f"{p}.{ff}.pre_layer_norm.weight"]
            sd[f"layers.{i}.{ff}.post_layer_norm.weight"] = raw[f"{p}.{ff}.post_layer_norm.weight"]
        for nm in ("norm_pre_attn", "norm_post_attn", "norm_out"):
            sd[f"layers.{i}.{nm}.weight"] = raw[f"{p}.{nm}.weight"]
        a = f"{p}.self_attn"
        per_dim = F.softplus(raw[f"{a}.per_dim_scale"]) * q_scale  # [128], applied per head dim, shared by heads
        clipped(f"layers.{i}.self_attn.q_proj", f"{a}.q_proj", per_dim.repeat(cfg.n_heads))
        clipped(f"layers.{i}.self_attn.k_proj", f"{a}.k_proj", k_scale)
        clipped(f"layers.{i}.self_attn.v_proj", f"{a}.v_proj")
        clipped(f"layers.{i}.self_attn.post", f"{a}.post")
        sd[f"layers.{i}.self_attn.relk"] = relative_keys(cfg, raw[f"{a}.relative_k_proj.weight"])
        c = f"{p}.lconv1d"
        sd[f"layers.{i}.lconv1d.pre_layer_norm.weight"] = raw[f"{c}.pre_layer_norm.weight"]
        clipped(f"layers.{i}.lconv1d.linear_start", f"{c}.linear_start")
        sd[f"layers.{i}.lconv1d.depthwise_conv1d.weight"] = raw[f"{c}.depthwise_conv1d.weight"]
        sd[f"layers.{i}.lconv1d.conv_norm.weight"] = raw[f"{c}.conv_norm.weight"]
        clipped(f"layers.{i}.lconv1d.linear_end", f"{c}.linear_end")
    sd["output_proj.weight"] = raw["audio_tower.output_proj.weight"]
    sd["output_proj.bias"] = raw["audio_tower.output_proj.bias"]
    sd["embedding_projection.weight"] = raw["embed_audio.embedding_projection.weight"]
    sd = {k: v.to(dtype) for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not any(s in k for s in ("sqrt_eps16", "floor16", "eps16", "res_w16", "inv_softcap16",
                                                               "softcap16", "mask_value", "one"))]
    assert not missing and not unexpected, (missing, unexpected)
    return m.to(dtype).eval()
