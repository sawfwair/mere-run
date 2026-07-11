# Guarded Acceleration Audit

This audit separates acceleration paths that preserve model semantics from
paths that change numerical precision, require checkpoint metadata, or only
move work around. A path is not promoted automatically merely because it is
theoretically faster.

## Shippable paths

### Shared autoregressive pipeline saturation

The shared decode loop confirms the first sampled token before opening its
deeper steady-state pipeline, then queues the next dependent model forward
before the host confirms the preceding token. The final token-budget boundary
samples without scheduling an unused forward. This keeps immediate EOS and
short budgets bounded without draining the GPU queue between ordinary tokens.

A matched release-mode gate on 2026-07-11 used the managed
`LiquidAI/LFM2.5-8B-A1B-MLX-8bit@984aa3f` checkpoint, a fixed prompt, greedy
sampling, and 96 output tokens on an M4 Max with 128 GB unified memory. The
three-run median was 63.25 decode tokens/s versus 62.47 for the pre-change
baseline (+1.2%). Peak physical footprint was 9.61 GB versus 9.79 GB (-1.8%),
and generated output was identical.

### Broader resident KV compression

`RuntimeKVCacheMode.affine8` selects groupwise affine 8-bit attention K/V for
Gemma4, Qwen-family full-attention layers, and LFM2 full-attention layers. Gemma
uses its existing model-specific quantized cache path. Qwen hybrid
linear-attention state and LFM2 convolution state keep their native
representations.

The generic Qwen/LFM2 cache supports prefix-cache forks, same-offset decode
batching, and row splitting. It dequantizes for attention because those engines
do not have a packed KV-attention kernel. Consequently affine 8-bit is an
explicit memory/quality control relative to full-precision KV, not an automatic
speed policy. Structural tests compare materialized packed bytes with BF16 and
numerical tests bound the 8-bit round-trip error.

For the generic Qwen/LFM2 cache, BF16 K/V with group size 64 versus packed
weights plus affine scales and biases yields 53.125% of the full-precision
resident bytes at equal allocation capacity: a 46.875% persistent-cache
reduction. Temporary dequantized attention inputs are not counted as resident
cache savings. This ratio does not apply to `text-chat-gemma4-turbo`: its
engine default is already a smaller 4-bit TurboQuant cache, so forcing affine
8-bit can increase KV residency.

```bash
swift run mere.run model runtime set text-chat-q36-nano \
  --kv-cache-mode affine8
```

Use `default` for the selected engine/model/server baseline; it does not promise
native full precision.

Real-checkpoint gate (2026-07-11, M4 Max 128 GB,
`LiquidAI/LFM2.5-8B-A1B-MLX-8bit@984aa3f`): after one warmup per mode, a
450-token prompt plus 24 greedy decode tokens was byte-identical. Default versus
affine-8 measured 0.629-0.692s versus 0.644-0.698s prefill and 0.369-0.397s
versus 0.407-0.457s decode across two gated post-warmup debug runs. This
supports the memory-control positioning, not a speed default: prefill was
effectively neutral and affine-8 decode was 10-15% slower on this short-context
case.

The same gate passed on
`mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit@63d5206`: 477 prompt tokens and 24
greedy tokens were byte-identical after per-mode warmups. Across two runs,
default versus affine-8 measured 0.383s versus 0.339-0.346s prefill and
0.898-0.918s versus 0.956-0.978s decode. That mixed result likewise argues for
an explicit memory mode: prefill improved 10-12% while decode was 6-7% slower.

### Psi compressed MLA

The native Psi/GLM attention path produces a shared KV latent, but the baseline
expands that latent into per-head keys and values before caching it. The guarded
path absorbs the key projection into queries and applies the value projection
after latent attention, retaining only the shared latent and decoupled RoPE key.

For representative dimensions (32 heads, rank 512, 128 no-PE dimensions, 64
RoPE dimensions, and 128 value dimensions), resident elements per token per
layer fall from 10,240 to 1,088, a 9.4x structural reduction. Weight absorption
reassociates floating-point operations and keeps dequantized projection weights
resident, so the default remains off pending a real-checkpoint quality corpus.

```bash
# Force for an A/B run.
MERERUN_PSI_COMPRESSED_MLA=1 swift run mere.run text chat \
  --model text-chat-psi-agent --temperature 0 --prompt "..."

# Explicit adaptive mode; still operator-selected.
MERERUN_PSI_COMPRESSED_MLA=auto \
MERERUN_PSI_COMPRESSED_MLA_MIN_PROMPT_TOKENS=2048 \
  swift run mere.run text chat --model text-chat-psi-agent --prompt "..."
```

### Psi sparse-MoE dispatch

The Psi expert loader now starts with packed placeholder tensors instead of
constructing enormous dense random expert tensors that are immediately
replaced by quantized checkpoint arrays. Long routed batches sort expert rows
before gather matmul. `MERERUN_PSI_FUSED_MOE=1` additionally stacks gate/up
expert rows so each block issues one gather matmul instead of two. Stacking is
numerically equivalent for per-row affine quantization, but retains a second
gate/up weight copy, so it is opt-in until a real checkpoint shows a worthwhile
end-to-end gain.

## Already covered

- Gemma4 12B uses a managed verifier-compatible MTP assistant and falls back to
  exact prompt-lookup speculation for repetitive greedy generation.
- Qwen3.6 Nano consumes its checkpoint MTP head only past the measured
  long-context break-even threshold.
- FLUX.2 Klein, HiDream O1, Z-Image Turbo, ACE-Step Turbo, Woosh DFlow, and LTX
  select distilled schedules from checkpoint or manifest variants. Base
  checkpoints retain their base schedules.

## Not honestly promotable

| Candidate | Current blocker |
| --- | --- |
| Activation quantization | MLX exposes quantized weight matmul, not a native activation-quantized attention/MoE contract. Quantize-then-dequantize would add kernels and quality loss without reducing the following operation. |
| MTP on every text model | Speculation requires a compatible draft head/model and verifier integration. Model metadata such as a next-token-layer count is not sufficient; unrecognized draft weights are not silently loaded. |
| Distilled steps on base diffusion checkpoints | Step distillation is checkpoint training, not a scheduler shortcut. Applying four-step schedules to base weights is a quality regression. |
| Compressed MLA outside Psi | Qwen and Gemma use GQA/hybrid attention rather than an MLA latent. The DeepSeek V4 Flash managed lane is an external native binary, so its internal cache cannot be changed in this Swift runtime. |
| Automatic Psi MLA/fused-MoE | Structural and algebraic tests pass, but no public Psi checkpoint is managed for repeatable output-quality and throughput A/B. Both controls remain explicit. |

## Promotion gate

Run greedy output parity and sampled distribution/quality evaluation on the
same model revision, prompt corpus, seed, and token budget. Report resident
memory, prefill tokens/s, decode tokens/s, and time to first token in release
mode. A structural reduction alone is not evidence of an end-to-end speedup.

Focused structural and numerical checks:

```bash
swift test --filter 'AffineQuantizedKVCacheTests|GLM47AccelerationTests'
```

When the managed LFM2 checkpoint is installed, run the gated real-checkpoint
greedy parity A/B on a GPU device:

```bash
MERERUN_TEST_GUARDED_ACCELERATION_CHECKPOINTS=1 \
MERERUN_TEST_MLX_DEVICE=gpu \
  swift test --filter GuardedAccelerationCheckpointTests
```
