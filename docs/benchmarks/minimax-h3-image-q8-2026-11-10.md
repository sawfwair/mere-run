# MiniMax-H3-Image native Q8 qualification

This report records the original generation-17 fallback measurements. The
[M5 Q8 acceleration follow-up](minimax-h3-image-m5-nax-2026-11-10.md) restores
scoped Q8 acceleration and qualifies subsequent attention and decode changes.

Native Swift/MLX Q8 export combines affine 8-bit/group-64 linear weights with
an exact six-step AdaLN cache. The original checkpoint remains separate.
Measurements use the same Apple M5 Max with 40 GPU cores, 128 GiB of unified
memory, macOS 27.2, AC power, camera prompt, and seed 42 as the
[original BF16 qualification](minimax-h3-image-native-qualification-2026-11-10.md).
Every image uses all six released PDD evaluations at CFG 1.

## Checkpoint

The source is
[MiniMax-H3-Image at the pinned revision](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/tree/492fd8c4eca666f1085f57e30738c1e0d86b3439).
The exported checkpoint occupies 52,035,131,360 bytes, or 52.04 GB / 48.46 GiB,
compared with roughly 135.38 GB for the selected original files. This reduction
combines three changes; it is not a measurement of quantization alone:

- Quantize eligible BF16 linear weights to MLX affine Q8/group 64. Deinterleave
  original per-head QKV rows before packing. Retain embeddings, FP32 tensors,
  and vision matrices whose input widths do not align to 64.
- Replace 26.02 GB of timestep-only block AdaLN weights with a 116.51 MB cache
  containing the original-precision outputs for every released timestep and
  the fixed reference timestep.
- Omit Qwen decoder layers after the layer-50 activation used for conditioning.

The transformer directory occupies 21.32 GB, the text encoder 28.22 GB, and
original FP32 VAE 2.34 GB. The export includes the tokenizer, processor, original
license and acknowledgment, source identity, and hashes of 24 weight shards.
`h3_image_mlx.json` distinguishes these native packed keys from raw checkpoint
keys. The loader validates all shards and required parameters.

## Optimizations from the upstream comparison

The [SGLang recipe](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3-Image.mdx)
precomputes AdaLN, prepares rotary tables and text refinement once per request,
retains model components, and caches conditioning for repeated inputs. The
native Q8 path now supports those operations for text-only benchmarks.
Reference conditioning is recomputed for every request.

The comparison also exposed a precision mismatch. Native rotary tables were
FP32, which promoted BF16 Q/K attention and the following residual stream to
FP32. The
[pinned reference implementation](https://github.com/huggingface/diffusers/blob/f598e6e6a99528858b962c1399c45b59a6c25484/src/diffusers/models/transformers/transformer_minimax_h3_image.py)
casts cosine and sine to the activation dtype before multiplication. The image
runtime now does the same. FP32 and mixed-BF16 six-step fixtures pass on CPU
and Metal after this correction.

Large sequences retain full attention to all keys while bounding projection
intermediates to 4096 rows and query batches to 640 rows. Contributor benchmarks
can change the number of heads per kernel. They can also expand Q8 transformer
weights to resident BF16 one block at a time, testing dense compute against
packed Q8 compute. This preserves Q8 storage but uses more runtime memory and
rounds the reconstructed weights to BF16; it does not recover original weights.

## Measurement boundaries

Release measurements include prompt preparation, all six denoising evaluations,
VAE decoding, and PNG encoding and output. The first request loads models and
conditioning; later requests reuse the transformer, VAE, and identical text-only
conditioning. OS peak process footprint from `/usr/bin/time -l` covers the whole
session. It is not a per-request MLX allocation counter or a minimum RAM claim.
File caches are not purged. Measurements use a shared desktop host.
The repeated requests use one identical prompt. A new prompt needs conditioner
execution while retained components remain, so its latency and memory can be
higher; that case is not measured here.

SGLang reports 16.21 seconds at 2048 × 2048 and 84.96 seconds at 4736 × 3552
on one DGX Spark GB10, with native precision, resident components, exact AdaLN
precomputation, and warm conditioning.
Its median covers 10 warm requests across two fresh servers and excludes startup,
HTTP, PNG encoding, and saving. Our square 4096 × 4096 canvas has a similar pixel
count but a different aspect ratio, and the hardware and quantization differ.
MLX affine Q8 is not CUDA FP8 or MXFP8. These figures cannot establish equal
hardware performance or a 70-second Apple Silicon target.

## Results

| Output and compute | First request seconds | Warm request seconds | Peak process footprint (GiB) |
| --- | ---: | ---: | ---: |
| 2K original BF16 baseline | 212.91 | — | 67.35 |
| 2K packed Q8 | 169.46 | — | 26.11 |
| 2K Q8 stored, resident BF16 compute | 58.05 | 43.99 | 40.93 |
| 4K original BF16 baseline | 498.93 | — | 72.12 |
| 4K packed Q8 | 510.76 | — | 26.16 |
| 4K Q8 stored, resident BF16 compute | 289.17 | 246.51 | 43.47 |

The CLI rows include admission; the retained contributor benchmark measures the
generator directly. Each dense row covers one first request and one repeated
warm request, rather than a throughput distribution. First-request file reads
vary with the OS cache. The 2K CLI run overlapped repository validation, which
finished during the 2K dense session. Our image inference runs were sequential.
The original BF16 baseline predates the cache and rotary correction, so these
are combined-runtime comparisons, not an isolated quantization speedup.

Packed Q8 2K denoising took 125.73 seconds, with the five later steps taking
about 18 seconds each. Dense BF16 compute from Q8-stored weights took 41.07
seconds for the first request's denoising loop and 39.32 seconds for the warm
request's loop.
The two dense 2K PNGs have identical SHA-256 hashes.

Packed Q8 4K denoising took 485.18 seconds, against 424.66 seconds in the
original BF16 baseline. Its five later denoising steps averaged 82.11 seconds.
The packed path substantially reduces memory here but does not improve 4K
wall time. Its five later steps spend 102.92 seconds in QKV preparation,
214.09 seconds in output/feed-forward work, and 93.47 seconds in attention.

With resident BF16 compute and 56-head attention batches, 4K denoising took
229.60 seconds for the first request and 228.88 seconds for the warm request.
Both 4K PNGs are byte-identical. The warm request spends 44.47 seconds in QKV
preparation, 92.57 seconds in output/feed-forward work, and 91.47 seconds in
attention. The remaining gap to the published GB10 pipeline time therefore
extends beyond loading and PNG output; denoising remains the dominant cost.
This qualification does not reach a 70-second native 4K result on the M5 Max.

The installed MLX fork explains an important part of this result. Its
[quantized dispatcher](https://github.com/sawfwair/mlx/blob/e3214809fa5ebd5aac5d6783e8101513b3d9a774/mlx/backend/metal/quantized.cpp#L21)
permits NAX matrix acceleration only on architecture generation 18 or later,
retaining a generation-17 correctness workaround. This M5 Max reports
`applegpu_g17s`. Floating-point matmul and attention retain NAX availability.
The qualification keeps the existing restriction; it does not bypass the
quantized regression guard. Results apply to the pinned MLX Swift revision
`1e077ba968b291e0237d1ab522b271d433fa1713` and embedded MLX revision
`e3214809fa5ebd5aac5d6783e8101513b3d9a774`.

An isolated BF16-input projection at the real feed-forward shape
`4096 × 5376 × 28672` measured a warm median of 91.34 ms for packed Q8 and
21.18 ms for dense BF16, excluding the first evaluation in each mode. This
4.3× kernel difference is not an end-to-end speed claim. It explains why lower
storage precision alone does not provide the fastest compute path on this M5.
A quiet attention probe favored 56-head batches over eight-head batches by
about 1.3×; increasing query batches beyond 640 rows did not improve its median.
Earlier probes overlapping image inference showed larger, unrepresentative
head-batching differences and are excluded from these conclusions.

### Visual review

The packed 2K image has a coherent brass rangefinder-style camera, one primary
lens, textured black covering, a blue cup behind on the left, and a walnut
tabletop. Whole-frame review and unscaled 512-pixel lens, covering, and tabletop
crops found no obvious tile seams or mosaic corruption. Dense BF16 compute
preserves the composition but changes small controls and invented markings.
The oversized cup and malformed engraving remain model limitations.

![Measured packed Q8 2K camera output, downsampled for this page.](/qualification/minimax-h3-image-q8-2026-11-10/camera-2048-packed-q8.jpg)

![Unscaled lens, covering, and tabletop crops from packed Q8 2K output.](/qualification/minimax-h3-image-q8-2026-11-10/camera-2048-packed-q8-details.jpg)

The packed 4K image retains fine lens rings, worn metal, and textured covering.
The reviewed frame and unscaled crops have no obvious tile seams or mosaic
corruption. Its composition places the subjects lower in a larger field of
blue background. Invented markings and the oversized cup remain.

![Measured packed Q8 4K camera output, downsampled for this page.](/qualification/minimax-h3-image-q8-2026-11-10/camera-4096-packed-q8.jpg)

![Unscaled lens, covering, and tabletop crops from packed Q8 4K output.](/qualification/minimax-h3-image-q8-2026-11-10/camera-4096-packed-q8-details.jpg)

The faster 4K mode retains the same main composition and material detail.
Its small controls and lens engraving differ from the packed path. Whole-frame
review and unscaled lens, covering, and tabletop crops found no obvious tile
seams or mosaic corruption. Equal hashes confirm identical output for the two
repeated requests in this session, not general visual equivalence across modes.

![Measured 4K output from Q8-stored weights with BF16 compute.](/qualification/minimax-h3-image-q8-2026-11-10/camera-4096-dense-bf16.jpg)

![Unscaled detail crops from the faster 4K mode.](/qualification/minimax-h3-image-q8-2026-11-10/camera-4096-dense-bf16-details.jpg)

## Reproduction

After installing and accepting the source checkpoint license:

```bash
mere.run model optimize image-minimax-h3 --output ./image-minimax-h3-q8 --json
mere.run model benchmark h3-image --model-root ./image-minimax-h3-q8 \
  --prompt "$PROMPT" --size 4096 --runs 3 \
  --output ./q8-benchmark --json
```

The exact prompt is retained with the measurement receipt and matches the
original BF16 report. Add `--resident-bf16` for dense transformer compute and
`--attention-heads 56` to test the larger attention batch. Set
`MERERUN_H3_IMAGE_PROFILE=1` for per-block and large-sequence QKV, attention,
and feed-forward timings on stderr. Each benchmark run retains its PNG and
SHA-256 hash. Use a new output directory for each configuration.

## Validation and remaining targets

`./scripts/check.sh` passed after the final Swift changes. Six image computation
fixtures and four Q8 packing/loading tests passed directly on Metal with
`MLX_ENABLE_TF32=0`; the repository gate covers their CPU paths. Generated
command documentation includes the new benchmark. The
[JSON receipt](/qualification/minimax-h3-image-q8-2026-11-10.json) retains binary,
source, shader-library, manifest, cache, and output identities, along with phase
timings and the standalone probe results.
The final 31 guide and documentation-contract tests and `pnpm docs:build`
also passed after the measured results were added.

The first real Q8 loader attempt exposed sparse module-array replacement across
shards. Updating a complete leaf-module tree preserves the other layers; a
regression loads array layers out of order, and the subsequent real generations
use the corrected loader. Failed and interrupted diagnostic runs are excluded
from the result table.

The next memory/speed experiment is temporary BF16 expansion of one transformer
block at a time, followed by release of its dense copy. The measured projection
gap motivates that experiment, but it is not implemented or timed here. It
would need comparison against both packed Q8 and resident BF16 for numerical
behavior and peak memory. Upstream's fused norm/rotary operations and batched
VAE tiles are additional targets. Native decode currently retains the original
FP32 VAE and sequential overlapping tiles; changing decoder precision also
needs separate quality qualification.

## Scope

This qualification covers one text-only prompt and seed at the measured sizes.
It does not establish general visual parity, reference editing fidelity,
typography accuracy, other seeds or aspect ratios, or support on smaller Macs.
Native MLX and PyTorch random streams differ, so equal integer seeds do not
produce a matched upstream image for pixel comparison.
