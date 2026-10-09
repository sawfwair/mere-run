# Clef Omni native Q4 qualification — 2026-10-09

This artifact packs the routed experts of
`Cloudflare/clef-omni@0db1cd2607d76a7bdb2a382f659e7b313079f84b`
with MLX affine 4-bit quantization, group size 64. Attention, routers,
embeddings, vision/audio towers, and the joint head retain BF16. The native
runtime omits upstream talker/code2wav weights.

## Reproducible conversion

Use `scripts/model-conversion/convert_clef_omni_mlx.py` on a CUDA MLX host.
The converter checks source shard hashes against the pinned Hub revision,
streams shards, and writes a typed quantization descriptor and a SHA256
manifest. Logical thinker weights total **21,763,806,432 bytes**; the bundle
before its conversion manifest totals **22,006,481,387 bytes**. These sizes
alone do not establish runtime memory fit.

The conversion used MLX 0.32.2. Twenty-four sampled expert tensors had relative
reconstruction MSE between 0.00844 and 0.00956. Weight reconstruction diagnostics
are separate from decision-quality measurements.

## Independent checkpoint controls

`scripts/reference-parity/qualify_clef_omni.py` loads the original BF16 thinker
with Transformers 5.10.2 and PyTorch 2.8.0, then evaluates eight synthetic probes:
invoice routing, structured JSON, Spanish routing, urgency scoring, a solid
color image, a one-second pure tone, a two-frame color video, and mixed
image/audio. The joint-head reference source has SHA256
`21d05ebb8cbea26a26af65eca3d16cf4b69bf80a320e2d5b203e86f0d632a670`.

A second control independently unpacks the quantized matrices into BF16
PyTorch expert parameters. All categorical decisions match the original BF16
run. The largest probability difference is **0.0035** across these probes.
Native tokenizer/media-placeholder IDs exactly match the original reference
for all eight cases.

The two-frame video probe is an agreement diagnostic: the BF16 reference
answers red even though the second supplied frame is blue. Agreement on that
probe must not be presented as correct temporal understanding. These probes
are not a general capability or accuracy benchmark.

## Mac execution

All eight native Metal probes passed on the 36 GB Mac (38,654,705,664 physical
bytes). MLX peak allocation was **22,063,592,210 bytes** (20.55 GiB), including
text weights and both media towers. The sequence completed in 45.92 seconds,
including loading. Inputs ranged from 151 to 294 tokens. All categorical choices
match BF16; maximum probability difference is **0.0036**, and urgency-score
difference is 0.0003. Against independently reconstructed Q4, maximum probability
difference is 0.0031.

The public CLI route probe also passed through normal shared-machine admission,
returning billing with probability 0.9873. Its process peak memory footprint was
20,011,421,128 bytes; this differs from MLX's allocator measurement. An earlier
attempt was correctly refused at 15.78 GB of reclaimable memory. The successful
run used the existing admission policy without overriding memory limits.
All 22 artifact files passed local SHA256 verification. See the adjacent
[receipt](./clef-omni-q4-qualification-2026-10-09.json) for hashes and raw decisions.

The opt-in `ClefOmniCheckpointParityTests` records case outputs, probability
differences, token counts, elapsed time, and MLX peak allocation.
Use a fresh process and an otherwise idle inference slot:

```sh
MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_TEST_CLEF_OMNI_ROOT=/path/to/clef-omni-mlx-q4 \
MERERUN_TEST_CLEF_OMNI_PARITY_DIR=/path/to/probes \
swift test --filter ClefOmniCheckpointParityTests
```

A pass establishes only the tested input sizes and media. It does not qualify
8,192- or 64,000-token contexts, maximum media budgets, file-video decoder
parity, Linux native execution, or broad decision accuracy. The artifact is
local; it has no published managed Q4 model ID.
