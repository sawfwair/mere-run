# Nemotron 3.5 Bolt

NVIDIA Nemotron 3.5 Bolt 74B-A9B-24GB is a hybrid Mamba, attention, and
mixture-of-experts model for text and image chat. `mere.run` loads the original
packed safetensors directly in Swift/MLX, so you don't need Python, vLLM, or an
offline conversion. Use it when you want a large sparse model whose weights fit
in about 24 GB.

## Install

The managed install is explicit; inference commands don't download it.

```bash
mere.run model pull vision-chat-nemotron-35-bolt-24gb
```

The original weights occupy 24,079,678,824 bytes. Tokenizer files, recurrent
state, image encoding, attention caches, and execution buffers add to that, so
the model name doesn't guarantee that the whole process fits in 24 GB. The
catalog lists 32 GB of unified memory as the minimum and 48 GB as recommended,
and shared admission checks still apply.

## Chat

Send a text prompt, or attach a local image:

```bash
mere.run text chat --model vision-chat-nemotron-35-bolt-24gb --prompt "Explain a Bloom filter."
mere.run text chat --model vision-chat-nemotron-35-bolt-24gb --image ./diagram.png --prompt "Describe this diagram."
```

Text and local image inputs, streamed generation, function tools, thinking
controls, stop sequences, and log probabilities use the shared chat path.

## Architecture

The runtime implements 40 Mamba-2 blocks, eight attention blocks without rotary
embeddings, and 40 heterogeneous latent MoE blocks. Each MoE block selects nine
experts. The checkpoint combines packed IQ2_XXS expert weights, packed Q8_0
attention and latent projections, NVFP4 Mamba projections, and BF16 embeddings
and vision weights. Packed weights stay in their original formats during
execution.

## Limits

Audio input, video input, LoRA, constrained JSON, and LiLiCorr speculative
decoding are unsupported. The source model's video capability isn't an
execution claim for this integration.

The [released-checkpoint diagnostic run](../qualification/nemotron35-bolt-checkpoint-2026-10-15.md)
passed 8/8 text and 4/4 image probes on a 36 GiB M3 Max. The report separates
debug-harness measurements, optimized CLI measurements, and peak process
footprint. These synthetic probes don't establish broad capability, upstream
parity, or long-context behavior.

## Run checkpoint diagnostics

Keep weights, fixtures, and raw results on a volume with room for the full
checkpoint. The following example uses an external model store and download
cache:

```bash
export MERERUN_MODELS_DIR=/Volumes/T7/bolt/models
mere.run model pull vision-chat-nemotron-35-bolt-24gb --cache-dir /Volumes/T7/bolt/hub-cache
swift scripts/validation/prepare_nemotron_bolt_suite.swift /Volumes/T7/bolt/fixtures
MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_TEST_BOLT_ROOT="$MERERUN_MODELS_DIR/vision-chat-nemotron-35-bolt-24gb" \
MERERUN_TEST_BOLT_SUITE=/Volumes/T7/bolt/fixtures/manifest.json \
MERERUN_TEST_BOLT_OUTPUT=/Volumes/T7/bolt/results \
/usr/bin/time -l swift test --filter NemotronBoltCheckpointTests
python3 scripts/validation/summarize_nemotron_bolt.py \
  /Volumes/T7/bolt/results --output /Volumes/T7/bolt/summary.json
```

To run a subset, set `MERERUN_TEST_BOLT_CASES` to comma-separated case IDs, such
as `smoke,ocr`.

The harness acquires the shared inference admission lease. During active
inference, the runtime also requests GPU residency through MLX's shared memory
manager. That request is capped below Metal's recommended working set and ends
with the request. The harness uses greedy decoding, a 4,096-token context limit,
and the original checkpoint's text and image path. The suite contains small
synthetic probes with exact expected answers and two capped decode throughput
runs.

Every completed case records its raw answer, prompt and generated token counts,
load, prefill, and decode times, wall time to the first visible piece, and peak
memory. These measurements have different scopes, so report them separately:

- MLX allocation peaks reset for each case and include resident weights.
- Process peak RSS and the kernel peak physical footprint are cumulative across
  the test process, including loading and earlier cases.
- The default SwiftPM harness is a debug build. Label optimized CLI
  measurements separately when you report throughput.

Exact-answer results on these probes don't establish broad capability, upstream
implementation parity, or long-context quality.

## Sources and license

The upstream checkpoint is pinned to
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB@96a5d63027ac013c8ef37fb438f3b4ec5fbd3ba6`.
Review the [OpenMDW-1.1 terms](https://openmdw.ai/license/1-1/) for the model
materials. The repository doesn't bundle the checkpoint weights.

For other text and vision chat models, see the [text runtime](./text.md).
