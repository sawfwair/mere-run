# Muse Glimmer

Native Swift/MLX text-and-image inference for Meta Muse Glimmer 30B.

The audited converter supports Q4 and Q8 and defaults to a selective Q4 layout that leaves the token
embedding and complete vision tower in BF16. A smaller `compact` scope is
available for explicit low-memory experiments. The loader discovers quantized
modules from the checkpoint's `.scales` arrays and accepts both upstream and
common MLX Hub key layouts, so selective and compact artifacts share the same
native runtime.

Image preparation performs separable uint8 Lanczos-3 resizing with per-pass
rounding and clamping before normalization, matching the released Pillow and
torchvision path. Vision position geometry stays in float32. Exact byte and bit
pattern fixtures cover both contracts so a visually plausible preprocessing
change cannot silently alter model input.

The patch projection uses its floating weight or quantization-scale dtype.
Bilinear position accumulation and rotary trigonometry stay in float32 before
their results are cast to the activation dtype. This keeps BF16 image features,
language embeddings, and KV state from being promoted to float32.

- `MuseGlimmerConfig.swift`: typed upstream configuration, including per-layer
  NoPE/rotary selection and optional MLX quantization metadata.
- `MuseGlimmerModel.swift`: dense local/global decoder, gated attention,
  centered residual norms, ViT-G/14 perception encoder, and multimodal scatter.
- `MuseGlimmerImageProcessor.swift`: aspect-preserving patch preparation and
  prompt placeholder expansion.
- `MuseGlimmerTokenizerAndTemplate.swift`: the released ATEM chat/tool template.
- `MuseGlimmerAssistantModel.swift`: five-layer DFlash and DFlash2 assistants,
  including target hidden-state projection and bidirectional sliding attention.
- `MuseGlimmerDFlash2.swift`: grouped dynamic causal convolution, the
  predecessor-aware sparse candidate selector, and published checkpoint-key
  normalization.
- `MuseGlimmerDFlashDecoder.swift`: target draft verification, rejection
  correction, and acceptance-aware target-only fallback.
- `MuseGlimmerGenerator.swift`: managed target/assistant loading and native decode.

The managed target is pinned to Sawfwair's selective MLX Q4 artifact at
`Sawfwair/Muse-Glimmer-30B-MLX-4bit@6532e898dc5c1a55b51b1b108cd36728b79be751`.
Its receipt pins the original Meta BF16 source at
`meta-models/Muse-Glimmer-30B@f84ecc3a0ea984a4c04542a84269e3d065350a6e`,
including every source and output hash. The automatically installed DFlash2
companion is pinned to
`z-lab/Muse-Glimmer-30B-DFlash2@b54ffdd11fa9cfe2af370012e5763d492c904128`.
The original companion at
`meta-models/Muse-Glimmer-30B-assistant@2c86316d689027b91123638739743fef1d425233`
remains a compatible fallback for existing installations.
Inference never invokes Python. The converters under `scripts/model-conversion/`
are offline release tooling only.

Native DFlash2 captures target layers 1, 13, 25, 37, and 49 during prefill,
projects them through the assistant, drafts from one real anchor plus mask
tokens, applies two-phase grouped dynamic causal convolution around every
attention and MLP sublayer, and scores sparse top-k candidates with a
predecessor-aware selector. The target verifies the entire candidate block in
one forward. Sampled rejection subtracts only the selector's sparse proposal
mass before drawing a replacement, preserving the target distribution. The
replacement becomes the next anchor without an extra recovery forward. The
sampled sparse path and exact greedy behavior on a deterministic test model
have end-to-end fixtures in `MuseGlimmerTests`.

Both checkpoints permit 15 draft tokens per round. The runtime retains the
measured Apple default of 3; `MERERUN_MUSE_GLIMMER_DFLASH_TOKENS` can override
it. For the original DFlash checkpoint, two interleaved 128-token M4 Max release
runs measured 14.73 tok/s target-only versus 20.07 tok/s DFlash at 54.9%
acceptance: 1.36x. That historical v1 diagnostic is not a DFlash2 result or a
cross-machine guarantee.

DFlash2 is the managed default. A matched real 64-token selective-Q4 probe
reached 17.01 tok/s with 67.7% acceptance versus 7.54 tok/s for one-token target
decode; its visible output was byte-identical to the existing DFlash v1 path,
which reached 63.1% acceptance on the same prompt. Both speculative paths share
MLX's pre-existing quantized block-verification rounding: at one near-tie,
one-token execution favored a token by 0.125 while batched verification rounded
the pair equally. This is not a new DFlash2 regression, but real-checkpoint
bit-exact equivalence to one-token execution is not claimed. Either assistant is
used only for output budgets of at least 32 tokens and returns to target-only
decode after two rounds below 40% acceptance.

Marked prefix checkpoints retain target KV and projected DFlash context together.
Restoration forks both states, so speculative decoding remains available after a
cache hit. Image KV is reusable when the actual resized pixels at that boundary
match. A later screenshot does not invalidate an earlier causal prefix. The
native prefix cache retains at most two entries and 65,536 tokens in total.

A separate scoped feature cache retains at most eight images and 8,192 image
feature tokens. It hashes the actual resized pixels, excludes images already
covered by restored KV, and clears with model unloading or replacement. Feature
reuse does not inflate the protocol's cache-read token count.

`MERERUN_MUSE_GLIMMER_PREFILL_CHUNK` selects a prefill chunk size from 128 to
2,048 tokens, clamped to that range; the default is 2,048. Cancellation is
checked between chunks. Use matched prompt and image workloads when tuning it.
The offline converter accepts `--bits 8` for selective Q8 with the same pinned
BF16 source verification, atomic output, and artifact receipt as Q4.
Q8 uses a symmetric affine encoding with an exact zero and codes fitted to the
stored BF16 scale. Standard affine BF16 scale rounding can shift zero; the
standard Q8 candidate failed its BF16 likelihood comparison. The symmetric
encoding retains the native packed format and BF16 activation range. It requires
no custom decode kernel. `--affine-method standard` reproduces the rejected Q8
recipe; `--affine-method auto` selects symmetric Q8 and preserves standard Q4.

To create a separate Q8 artifact without replacing the BF16 checkpoint, run:

```sh
uv run --script scripts/model-conversion/convert_muse_glimmer_mlx.py \
  --source /path/to/upstream-bf16 \
  --output /path/to/separate-glimmer-q8 \
  --bits 8
```

For a custom root, use its served directory-based ID in Messages requests and
`api connect claude-code --model`. Model discovery reports the resolved adapter's
capabilities and effective limits for that ID.

The November 8 M5 Max comparison scored 32,768 identical next-token positions.
BF16 perplexity was 5.9902, symmetric Q8 was 5.9926, and standard Q4 was 6.2016.
The paired Q8 increase was 0.04%; Q4 increased 3.53%. Both passed the predeclared
5% limit, including the upper 95% window-bootstrap bound. These are bounded text
diagnostics; see [Messages qualification](../../../docs/internals/messages-qualification.md)
for task evidence and limitations.
Q8 is selected for the 128 GB M5 Max comparison. The measurable Q4 regression
is not accepted for this setup, and further Q4 task qualification was stopped.
The existing Q4 conversion default remains unchanged for compatibility.

`api serve --warmup` includes managed and custom Glimmer roots. Startup timing
separates loading from warmup, and the server becomes ready after native target
and DFlash warmup. `--no-warmup` postpones that work to the first generation.

A paired 16-window, 8,192-token WikiText-2 candidate check measured 9.535
perplexity for selective Q4 and 9.554 for compact Q4. The selective-minus-compact
mean NLL delta was -0.0020 with a 95% bootstrap interval of -0.0035 to -0.0004.
That bounded sample supports selective Q4 as the default conversion scope; it is
not a substitute for a full release-quality evaluation.

The official BF16 assistant is preferred even beside a Q4 target. An optional
Q4 assistant measured slower at the same acceptance rate on this workload, so
its converter is retained for reproducibility rather than selected first.

On Metal generation 17 (M5), the pinned backend disables quantized NAX kernels
for correctness. Large affine-Q4/Q8 prefill matrices instead reconstruct a
transient dense weight and use dense BF16 NAX. Inputs below 128 rows, including
serial decode and DFlash verification, retain packed quantized execution.
`MERERUN_MUSE_GLIMMER_QUANTIZED_PREFILL` accepts `auto` (the default), `native`,
or `dense`. Automatic selection requires generation 17 and macOS 26.2 or later.
This preserves the backend safety guard and retains no second model copy.
Dense reconstruction and native quantized matmul can round differently; evaluate
quality with the selected execution path.

The opt-in checkpoint harness is
`MuseGlimmerCheckpointQualificationTests`. It records actual token sequences,
per-token NLL, predicted token IDs, prefill and decode times, thermal state,
and peak active MLX allocation. `scripts/reference-parity/compare_glimmer_quantization.py`
compares identical held-out windows and reports a paired bootstrap interval.
The float32 mathematical GPU fixtures use `MLX_ENABLE_TF32=0`; normal BF16 and
quantized performance measurements retain the default accelerated settings.
