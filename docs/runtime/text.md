# Text runtime

Use the text runtime to stream chat tokens, run a sandboxed tool loop, include
an image with a prompt, produce a grammar-checked JSON object, and fine-tune on
your own conversations. Separate paths provide GGUF code generation, embeddings
for local search, and local personally identifiable information (PII)
redaction.

## Commands

| Command | What it does |
| --- | --- |
| `mere.run text chat` | Run local chat with text chat models. |
| `mere.run text code` | Run local code generation with GGUF models via llama.cpp. |
| `mere.run text embed` | Generate text embeddings using native Qwen3-Embedding-0.6B. |
| `mere.run text anonymize` | Detect and redact PII using OpenAI Privacy Filter. |
| `mere.run text train-lora` | Train a native text LoRA adapter from chat-style SFT JSONL. |

## macOS Studio

The macOS app compiles against the same `MereRunContract` emitted by
`mere.run catalog --json`. Chat's Options panel exposes text versus constrained
`json_object` output, model-default/show/disable reasoning, context and sampling
controls, key-value (KV) quantization, a catalog adapter ID or local LoRA file, tool
permissions, preflight, and installed-model enforcement. **Utilities** turns
Embeddings into a vector explorer with dimensions, norms, previews, and cosine
similarity, and turns Anonymize into original/protected text review with labeled
PII spans. **Training** opens a text-dataset preview, preflight, live metrics,
artifacts, history, and comparison workspace for native LoRA. Advanced retains
guided raw forms; every generated flag is checked against public ArgumentParser
help in the repository gate.

## Model families

### Chat

- `text-chat-gemma4`
- `text-chat-gemma4-12b` (managed dense Google Gemma 4 12B-it snapshot)
- `text-chat-gemma4-12b-4bit` (managed MLX 4-bit Gemma 4 12B-it snapshot)
- `text-chat-gemma4-turbo` (managed MLX NVFP4 Gemma 4 26B-A4B MoE snapshot)
- `vision-chat-muse-glimmer-30b` (pinned Sawfwair selective MLX Q4 conversion of Meta Muse Glimmer 30B)
- `text-chat-laguna-s-2-1` (managed Poolside Laguna S 2.1 118B-A8B NVFP4 target plus DFlash)
- `text-chat-laguna-xs-2-1` (managed Poolside Laguna XS 2.1 33B-A3B NVFP4 target)
- `text-chat-nemotron-35-lightning` (managed NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 target plus DSpark)
- `text-chat-q36-nano`
- `vision-chat-q38-27b` (managed official Qwen3.8 27B BF16 vision-language snapshot)
- `vision-chat-q38-27b-4bit` (managed MLX 4-bit target plus pinned official MTP shard)
- `text-chat-bonsai-27b-1bit` (managed packed 1-bit dense Qwen3.6 27B vision/reasoning snapshot)
- `text-chat-bonsai-27b-2bit` (managed packed 2-bit ternary dense Qwen3.6 27B vision/reasoning snapshot)
- `text-chat-lfm25-2.6b-4bit` (managed LiquidAI LFM2.5 2.6B dense MLX 4-bit snapshot)
- `text-chat-lfm25-2.6b-qad-4bit` (managed QAD-trained LFM2.5 2.6B native MLX 4-bit conversion)
- `text-chat-lfm25-2.6b-bf16` (managed LiquidAI LFM2.5 2.6B BF16 target plus DSpark)
- `text-chat-lfm25-1.2b-bf16` (managed LiquidAI LFM2.5 1.2B Instruct BF16 target plus DSpark)
- `text-chat-lfm25-1.2b-qad-4bit` (managed QAD-trained LFM2.5 1.2B native MLX 4-bit conversion)
- `text-chat-lfm25-a1b-8bit` (managed LiquidAI LFM2.5 8B-A1B MLX 8-bit snapshot)
- `text-chat-lfm25-a1b-bf16` (managed LiquidAI LFM2.5 8B-A1B BF16 target plus DSpark)
- `vision-chat-lfm25-3b-8bit` (managed LiquidAI LFM2.5-VL 3B MLX 8-bit vision-language snapshot)
- `text-agent-ornith-9b` (experimental native MLX/OptiQ coding-agent snapshot)
- `text-agent-ornith-35b-mlx-4bit` (Ornith 1.5 speed tier)
- `text-agent-ornith-35b-mlx-6bit` (Ornith 1.5 balanced tier)
- `text-agent-ornith-35b-mlx-8bit` (Ornith 1.5 quantized quality tier)
- `text-agent-ornith-35b-mlx` (Ornith 1.5 35B-A3B BF16 MLX coding-agent snapshot)
- `text-agent-deepseek-v4-flash` (API/agent serving)
- `text-chat-mebot` (API serving; not a `text chat` dispatch lane)
- `text-chat-psi-agent`

### Code

- `text-agent-ornith-35b`
- `text-code-north-mini`
- `text-code-qwen3`

### Embeddings

- `text-embed-qwen3-0.6b`

### Anonymization

- `text-anonymize-privacy-filter`

## Typical workflows

### Local chat

```bash
swift run mere.run text chat \
  --stream \
  --model text-chat-gemma4 \
  --prompt "Summarize diffusion models in one paragraph."
```

`text-chat-gemma4-12b` runs Google's dense Gemma 4 12B-it checkpoint through
the native Swift Gemma runtime. Managed pulls for `text-chat-gemma4-12b` and
`vision-chat-gemma4-12b` install the `google/gemma-4-12B-it-assistant`
companion as `text-chat-gemma4-12b-mtp`; when it is present, greedy serial
decode can use native MTP on the decode tail after text or multimodal prefill.
Sampled requests, prefix-KV seeded requests, continuous batching, raw local
model paths, and prompts below the MTP threshold fall back to baseline decode.
When no assistant is active, greedy requests use draft-model-free
prompt-lookup speculation by default: repeated context n-grams are drafted
and verified in bursts inside the pipelined loop — output-identical, free on
non-repetitive text, and worth ~2× when generation echoes the prompt
(`MERERUN_GEMMA4_PROMPT_LOOKUP=0` disables; see
[Configuration](../configuration.md#mererun_gemma4_prompt_lookup)).

`vision-chat-muse-glimmer-30b` installs Sawfwair's pinned 21.38 GB selective MLX
Q4 target plus z-lab's DFlash2 assistant. The target's receipt pins every
source and output hash and retains Meta's bundled terms. Native DFlash captures
the five released target hidden-state taps during text or image prefill, drafts
masked blocks, and verifies them against the target. It engages for output
budgets of at least 32 tokens, defaults to 3 proposals per round on MLX, and
returns to target-only decode after two rounds below 40% acceptance. Add
`--stats` to see the proposal width, acceptance, verification forwards, and
fallback state.
Loading performs one target and DFlash warmup before the first user-visible
decode, so lazy MLX graph compilation is reported in `loadSeconds` and a
persistent CLI/API model session pays that cost once.

DFlash2 adds two-phase grouped dynamic causal convolution and predecessor-aware
sparse candidate selection. It is installed and preferred automatically; an
existing original DFlash or local Q4 assistant remains a fallback. A real
64-token selective-Q4 probe produced the same bytes as the original DFlash path
while improving acceptance from 63.1% to 67.7% on that prompt.

```bash
swift run mere.run model pull vision-chat-muse-glimmer-30b --accept-model-license
swift run mere.run text chat \
  --model vision-chat-muse-glimmer-30b \
  --prompt "Inspect this rollout plan for unsafe assumptions." \
  --stats
```

`text-chat-nemotron-35-lightning` is an explicit native Swift/MLX lane for
NVIDIA's 30B-total, 3B-active Nemotron-H model. The runtime implements its
52-layer Mamba-2, attention, and routed-MoE stack directly; it consumes the
released NVFP4 values without requantization and uses FP32 recurrent state for
the Mamba layers. Pulling the target also installs the converted 967M DSpark
companion:

```bash
swift run mere.run model pull text-chat-nemotron-35-lightning
swift run mere.run text chat \
  --model text-chat-nemotron-35-lightning \
  --temperature 1 \
  --top-p 0.95 \
  --stats \
  --prompt "Design a recovery-safe Swift actor pipeline."
```

DSpark uses NVIDIA's recommended three-token proposal width. Every proposal is
verified by the target; after two rounds below 67% measured acceptance, the
request continues on the faster serial target path. `--stats` reports the
proposal, verification, recovery, and adaptive-fallback counts. Set
`MERERUN_NEMOTRON35_DSPARK=0` to compare serial target decode, or tune the
break-even gate with `MERERUN_NEMOTRON35_DSPARK_MIN_ACCEPTANCE`. The model is
never selected or downloaded implicitly; 32 GB is the catalog floor and 64 GB
provides additional runtime headroom.

Use these recommended chat models by unified-memory band:

| Unified memory | Recommended model | Notes |
| --- | --- | --- |
| 16-23 GB | `text-chat-gemma4-12b-4bit` | Compact default; `text-chat-gemma4-nano` has the smallest memory requirement. |
| 24-63 GB | `text-chat-gemma4-12b-4bit` | Default grounded local-assistant tier; `text-chat-gemma4-turbo` is the larger Gemma alternate. |
| 64-95 GB | `text-chat-gemma4-12b-4bit` | Retain the measured Gemma assistant lane; use additional memory for context, concurrency, or larger Gemma alternatives. |
| 96+ GB | `text-agent-deepseek-v4-flash` | Agent and API chat tier; retain Gemma 12B 4-bit for interactive local chat. |

The DeepSeek V4 Flash tier uses the official 80.76 GiB pure-Q2 0731 imatrix
GGUF. Its persistent DS4 server defaults to a 32K operational context, a
1,024-token prefill chunk, and an 8 GiB disk-KV budget. A 128 GB Mac provides
additional headroom. Close other memory-heavy workloads before using the model
on a 96 GB machine.

`text-chat-lfm25-1.2b-bf16`, `text-chat-lfm25-2.6b-bf16`, and
`text-chat-lfm25-a1b-bf16` install exact pinned LiquidAI LFM2.5 checkpoints and
their matching DSpark assistants. All three run through the native Swift LFM2
runtime and are text-only. Select one with `--model`; `api serve --engine
text-chat-lfm2` defaults to the A1B model unless `--model` is provided.
DSpark uses five non-causal diffusion-attention layers to draft nine-token
blocks from five configured target-layer outputs and verifies them with the target before
committing output. Greedy tokens remain target-authoritative: accepted draft
prefixes are committed, while a mismatch partially rolls target attention and
short-convolution state back to the accepted prefix without replaying the
rejected block. Sampled generation uses rejection sampling, and the runtime
adaptively falls back if draft acceptance stays below 20% for three rounds.
`text chat --stats` prints `lfm25_dspark` state and acceptance counters.

The QAD ids use deterministic Sawfwair-hosted native-MLX conversions of
LiquidAI's QAD Q4_0 GGUFs. Projection nibbles and scales are preserved exactly
in affine group-32 tensors; the tied Q6_K embedding is requantized to MLX
6-bit/group-64 with error recorded in `MERERUN_CONVERSION.json`. QAD's benefit
is accuracy recovery at a compact Q4_0 footprint. It is not a second speedup
over Q4_0; MLX throughput depends on the native group-32 kernel. Use 1.2B QAD
as the memory-first tier, the standard 2.6B MLX checkpoint as the faster compact
2.6B tier, and 2.6B QAD when quantized quality recovery is the priority.

Set `MERERUN_LFM25_DSPARK=0` for a target-only A/B or
`MERERUN_LFM25_DSPARK_PATH` to load an explicit compatible sidecar. Very short
outputs, vision prefill, logprob capture, continuous batching, and target
prefix-cache reuse currently take the target-only path. Without DSpark,
concurrent serve workloads use ragged, cache-safe decode batching when
`--max-active-requests` is above `1` (`MERERUN_LFM2_CONTINUOUS_BATCHING`
overrides); rows at different prompt lengths share a forward only when every
attention and short-conv cache proves compatibility.

```bash
swift run mere.run model pull text-chat-lfm25-2.6b-bf16 --accept-model-license
swift run mere.run text chat \
  --model text-chat-lfm25-2.6b-bf16 \
  --stats \
  --prompt "Explain speculative decoding in one paragraph."
```

`vision-chat-lfm25-3b-8bit` adds the checkpoint's SigLIP2 vision tower and
multimodal projector to that native LFM2 engine. Pass a local image path or
base64 data URL through `--image`; the runtime smart-resizes the image to the
checkpoint's 16-pixel patch grid, projects the downsampled visual tokens into
the language prompt, then reuses the normal LFM2 decode path.

```bash
swift run mere.run model pull vision-chat-lfm25-3b-8bit --accept-model-license
swift run mere.run text chat \
  --model vision-chat-lfm25-3b-8bit \
  --image ./document.png \
  --prompt "Summarize this document and list its key figures."
```

`text-chat-laguna-s-2-1` is an opt-in 96 GB-and-up Apple Silicon lane and is
never selected by setup or hardware-aware defaults. Its roughly 72 GB target
and 2.2 GB DFlash companion use immutable official Poolside revisions. The
public OpenMDW-1.1 repositories are not gated:

```bash
swift run mere.run model pull text-chat-laguna-s-2-1
swift run mere.run text chat \
  --model text-chat-laguna-s-2-1 \
  --prompt "Implement a bounded Swift actor queue and explain its invariants." \
  --stats
```

The native runtime defaults Laguna to temperature `1`, top-p `1`, top-k `20`,
and the measured min-p `0.02`. Automatic DFlash proposes 12 tokens for output
budgets of at least 32 tokens and falls back losslessly to target-only decode
when acceptance is poor. Longer requests can use ragged continuous batching in
`api serve` when `--max-active-requests` is above `1`.

`text-chat-laguna-xs-2-1` is the smaller 36 GB-minimum / 48 GB-recommended
lane. It pins Poolside's five-shard `Laguna-XS-2.1-NVFP4-mlx` revision and
uses the same native runtime with the XS-validated M5 decode and terminal
prefill accelerations, traced back to Poolside's canonical released
`Laguna-XS-2.1-NVFP4` model. The public repository is not gated; the managed
entry retains its OpenMDW-1.1 license file, never auto-downloads because of its
size, and does not attach the S DFlash checkpoint:

```bash
swift run mere.run model pull text-chat-laguna-xs-2-1
swift run mere.run text chat \
  --model text-chat-laguna-xs-2-1 \
  --prompt "Implement a bounded Swift actor queue and explain its invariants." \
  --stats
```

`text-chat-inkling-small` installs mere.run's immutable native MLX conversion
of Thinking Machines Lab's released Inkling-Small checkpoint. The 256 routed
experts in each sparse layer use MLX affine 2-bit quantization with group size
128. Attention, embeddings, routers, shared experts, dense MLPs, and norms
retain the released BF16 precision. Because it is still a very large artifact,
it never auto-downloads from an inference command; pull it explicitly on a
128 GB unified-memory machine.

The released model is a 276B-total / 12B-active multimodal MoE with a
1,048,576-token architecture limit. This first mere.run lane loads only the
`language_model.*` weights in native Swift/MLX; image and audio towers are not
claimed here. The operational context defaults to
32,768 tokens to leave room for weights, KV cache, and macOS on a 128 GB Mac.
Larger `--context-size` values are accepted when the host has the additional
memory.

This mixed recipe replaces the rejected blanket 2-bit experiment, which was
degraded and repetitive even on a basic arithmetic prompt. Keeping every
non-routed path at BF16 deliberately spends more memory to protect model
quality. The managed snapshot is 84.56 GB (78.75 GiB). A full CUDA load used
84.53 GB of active MLX memory and peaked at 84.77 GB on an H200; greedy checks
correctly answered `2 + 2`, the "all but 9" sheep question, and generated a
valid Swift `isEven` function. That is artifact and CUDA-runtime proof; the
native Apple Silicon generation smoke remains a separate gate.

```bash
swift run mere.run model pull text-chat-inkling-small
swift run mere.run text chat \
  --model text-chat-inkling-small \
  --context-size 32768 \
  --prompt "Design a recovery-safe migration plan for a large Swift service." \
  --stats
```

Per-model API-serving KV behavior is set with `mere.run model runtime
--kv-cache-mode` (see [Model Management](./model-management.md)); it is not a
flag on `text chat` or `api serve`. For Gemma4, Qwen-family, and LFM2 models
you can select explicit `affine4` or `affine8` as long-context memory
controls relative to full-precision K/V. Qwen-family and LFM2 dequantize the
generic cache for attention, so these modes are not assumed faster. Gemma uses
its model-specific quantized KV path; `text-chat-gemma4-turbo` already defaults
to a smaller 4-bit TurboQuant cache, so forcing affine 8-bit can increase that
model's KV residency. `default` restores the engine/model/server default rather
than promising full precision.

`vision-chat-q38-27b` is the official dense Qwen3.8 27B BF16 checkpoint. Pull
it explicitly before use because the pinned snapshot is 55.59 GB:

```bash
swift run mere.run model pull vision-chat-q38-27b
swift run mere.run text chat \
  --model vision-chat-q38-27b \
  --image ./diagram.png \
  --prompt "Explain this diagram and verify every label."
```

The lane uses the published 262,144-token context, thinking default,
temperature 1.0, top-p 0.95, top-k 20, both generation stop tokens, and the
Qwen3.8 image sizing floor. Its native reasoning-effort values are `low`,
`medium`, and `xhigh`; Pi's `minimal`, `high`, and `max` aliases map to those
native values without inventing unsupported template labels. Tool calls use a
streaming structural Qwen XML parser, including when parameter strings contain
tag-like text. The checkpoint also contains video understanding weights, but
the native command accepts text and local images only. Its embedded
dense MTP head can be loaded from the official shards with
`MERERUN_Q35_MTP_SPECULATION=1`. This materially accelerates greedy decode, but
is experimental: BF16 multi-token verification can choose a different greedy
path from serial target decode. The default, sampled, and JSON-constrained paths
retain target-only decode. With API concurrency enabled, an eligible MTP request
uses speculation only while uncontended; peers use the continuous-batching lane.

Qwen3.8 prefill defaults to 1,024-token chunks. Live reclaimable host-memory
headroom below 16 GiB, or an admitted peer, caps a chunk at 512 tokens. This
uses live pressure instead of total physical memory because the dense BF16
checkpoint and concurrent workloads can consume most unified memory before
prefill begins. `MERERUN_Q35_PREFILL_CHUNK_TOKENS` remains an explicit upper
bound and is still pressure-capped.

For the lower-residency lane, pull the separate 4-bit model ID:

```bash
swift run mere.run model pull vision-chat-q38-27b-4bit
swift run mere.run text chat \
  --model vision-chat-q38-27b-4bit \
  --prompt "Implement a bounded async work queue in Swift."
```

This installs the pinned MLX Fast 4-bit/group-64 target and its matching
proposal-only 4-bit/group-64 MTP head under `mtp/`, totaling about 15.39 GB;
the managed install also retains Qwen's official Apache-2.0 license. On the
measured M4 Max coding slice,
target-only warm decode reached about 26.5 tok/s versus 8.8 tok/s for BF16;
explicit MTP reached 37.8–43.8 tok/s across the three short cases. All cases
passed. On a deterministic 24-task stride through the official HumanEval set,
target-only and explicit MTP both passed 20/24 with the same four failures, but
one failing case generated 174 tokens with MTP versus 177 target-only. MTP also
changed a thinking-mode token trajectory, so it remains explicitly opt-in and
is disabled for sampled and JSON-constrained generation.

`text-chat-bonsai-27b-1bit` and `text-chat-bonsai-27b-2bit` install the pinned
5.13 GB binary and 8.52 GB ternary Prism ML snapshots. They run packed low-bit
language weights plus a dense vision tower through the native Qwen-family
runtime. Both default to thinking-enabled generation and the published
temperature 0.7, top-p 0.95, and top-k 20 when those values are not set
explicitly. The models advertise 262,144 tokens; `text chat` selects that limit
automatically, and `--context-size` can set a smaller operational bound. Choose
1-bit for lower residency and faster decode or 2-bit for the larger ternary
checkpoint. On Linux CUDA, the pinned MLX fork executes supported 1-bit affine
projections directly from packed weights; the runtime retains its operation-
scoped dense fallback for unsupported shapes and keeps 2-bit and wider model
dispatch independent. For memory-constrained long-context work, opt into the
generic affine cache:

```bash
swift run mere.run model pull text-chat-bonsai-27b-1bit
swift run mere.run model pull text-chat-bonsai-27b-2bit
swift run mere.run text chat \
  --model text-chat-bonsai-27b-2bit \
  --context-size 262144 \
  --kv-bits 4 \
  --prompt "Summarize the key decisions in this context."
```

The local-path-only `text-chat-psi-agent` runtime has guarded compressed-MLA
and fused sparse-MoE controls. They remain opt-in until a repeatable public
checkpoint quality/throughput A/B is available; see the
[guarded acceleration audit](../internals/guarded-acceleration.md).

`text-agent-ornith-9b` installs an Ornith 1.0 9B OptiQ MLX snapshot and runs
through the native Qwen-family runtime. Use `text chat --model text-agent-ornith-9b`
or `api serve --engine text-chat-q36 --model text-agent-ornith-9b` for coding-agent
smoke tests.

Ornith 1.5 35B-A3B uses the same native Qwen-family lane across the official
`text-agent-ornith-35b-mlx-4bit`, `-6bit`, `-8bit`, and BF16
`text-agent-ornith-35b-mlx` targets. Runtime-triggered auto-download stays
disabled. `model capabilities` selects explicit speed/balanced/quality ids
from unified memory: Q4 starts at 32 GB, Q6 at 48 GB, Q8 at 64 GB, and BF16 at
96 GB. Pulling any tier also installs one shared MTP head from the pinned
authoritative base checkpoint; target verification remains authoritative for
every emitted speculative token. Ornith enables that verified MTP path from
short prompts; Qwen3.6 retains its separate 6,144-token adaptive threshold.

Both Ornith lanes are R1-style reasoning tunes and generate with thinking
enabled by default in `text chat` and `api serve` — without it the models
degenerate into repetition loops or signature echo on constrained prompts.
The reasoning stays hidden in `text chat` unless `--thinking` is passed;
`--no-thinking` disables reasoning generation entirely. When sampling options
are not set explicitly, these lanes also use the model's published
`generation_config.json` sampling (temperature 1.0, top-p 0.95, top-k 20)
instead of the generic chat defaults. Other Qwen-family lanes such as
`text-chat-q36-nano` keep the existing no-think default.

Native and llama.cpp text generation also accept min-p sampling. A value such
as `0.05` removes tokens with less than 5% of the most likely token's
probability after top-k/top-p filtering, then renormalizes the remaining
distribution. It is disabled by default except for Laguna's measured `0.02`,
does not change temperature-zero greedy output, and is available as `--min-p`
in text commands and `min_p` in OpenAI-compatible chat requests.

Native MLX Gemma and Qwen-family chat models support
`--response-format json_object`. The output must be one complete root object;
the prefix grammar checks nested objects and arrays, strings and escapes,
Unicode, numbers, booleans, null, commas, and colons before each token is
streamed. Qwen-family JSON mode forces thinking off and bypasses MTP,
continuous batching, and pipelined decoding. The GGUF Q36 model used by Linux
packages remains on llama.cpp and rejects JSON-object mode until a llama.cpp
grammar is wired.

```bash
swift run mere.run text chat \
  --model text-chat-q36-nano \
  --response-format json_object \
  --prompt 'Return an object with a name and an array of tags.'
```

### Tool use

`text chat` can run a local agentic tool loop. `--tools` takes a
comma-separated list of built-in tool names (`write_file`, `shell_exec`), and
`--tool-loop` enables the loop: generate, execute tool calls, feed results
back, continue, capped at 10 iterations. Tools execute inside a sandbox
directory — `--sandbox-dir` sets it; the default is a per-process temp
directory — and each call asks for interactive `[y/N]` approval by default.
`shell_exec` additionally requires `--allow-shell-exec`, and `write_file` only
targets absolute paths outside the sandbox with `--allow-absolute-tool-paths`.
`--auto-approve-tools` skips confirmation for `write_file` only; `shell_exec`
always requires interactive approval even when that flag is set.

```bash
swift run mere.run text chat \
  --model text-chat-gemma4-12b-4bit \
  --prompt "Create hello.py that prints the system time, then run it." \
  --tools write_file,shell_exec \
  --tool-loop \
  --allow-shell-exec \
  --sandbox-dir ./work
```

### Vision input

`text chat` accepts `--image` with a local file path or a `data:image/...` URI
for vision-capable chat models such as Bonsai 27B or `vision-chat-gemma4-12b`.
The image attaches to the user prompt for multimodal prefill.

```bash
swift run mere.run text chat \
  --model text-chat-bonsai-27b-2bit \
  --image ./photo.png \
  --prompt "Describe what is in this photo."
```

### Local code generation

```bash
swift run mere.run text code \
  --prompt "Write a Swift function that reverses a string."
```

`text-code-qwen3`, `text-code-north-mini`, and `text-agent-ornith-35b` are
native `text code` models and run GGUF weights through llama.cpp. North Mini
Code uses the Unsloth `North-Mini-Code-1.0-UD-Q4_K_M.gguf` quant, so it
requires a llama.cpp runtime with `cohere2moe` architecture support. Ornith 35B
uses DeepReinforce's `ornith-1.0-35b-Q4_K_M.gguf` quant for larger coding-agent
evals.

```bash
swift run mere.run model pull text-code-north-mini
swift run mere.run text code \
  --model text-code-north-mini \
  --prompt "Write a Swift function that reverses a string."

swift run mere.run model pull text-agent-ornith-35b
swift run mere.run text code \
  --model text-agent-ornith-35b \
  --prompt "Write a compact Swift Result helper."
```

### Embeddings

```bash
swift run mere.run text embed \
  "semantic search query" \
  --pretty
```

### PII anonymization

```bash
swift run mere.run text anonymize \
  "My name is Dana Example and my email is dana@example.com"
```

### Native text LoRA training

```bash
swift run mere.run text train-lora \
  --data ./pairs.seed.jsonl \
  --eval ./eval.prompts.jsonl \
  --output ./local-assistant.safetensors \
  --model text-chat-gemma4-12b-4bit \
  --dry-run \
  --json
```

`text train-lora` is the native MereRun entrypoint for chat-style SFT JSONL.
The data format is one JSON object per line with `sources`, `messages`, and an
optional typed `tools` array. Messages use the same `system`, `user`,
`assistant`, and `tool` roles as the local chat runtime. Assistant messages can
carry typed `toolCalls`; tool results use `name` and `toolCallID`. The selected
model's native template receives the same complete tool schemas during
training that it receives during inference, and native tool-call markup is
included in the assistant loss target. `--dry-run` validates the dataset,
fingerprints messages and complete tool schemas, validates and counts an
optional held-out SFT dataset, and writes a `.manifest.json` next to the
requested adapter path.

Duplicate detection compares the complete controller prefix and its tool
schemas, rather than only the first user question. A curriculum may therefore
repeat a question after an assistant tool call and tool result, while an exact
duplicate controller state remains invalid.

Without `--dry-run`, the command resolves a supported Gemma 4 text model,
`text-chat-laguna-xs-2-1`, or `text-chat-inkling-small` through the same managed
model store as chat, applies that family's released chat template, masks loss
to assistant tokens, injects native LoRA layers into the family-specific target
surface, and writes a `.safetensors` adapter plus a family-specific manifest. Add
`--visualize` to start the same loopback LoRA training dashboard used by image
training; text runs write `run.json`, `*.events.jsonl`, `*.loss.csv`, and
`*.loss.html` beside the adapter so loss and training events can be inspected
while the optimizer runs. Keep local text fine-tuning in `mere.run` so the same
model IDs, manifests, runtime constraints, and evaluation artifacts remain
under the MereRun command surface.

When `--eval` is supplied for a real training run, mere.run tokenizes that
held-out SFT JSONL with the same model chat template and reports assistant-token
negative log-likelihood immediately before and after optimization. The held-out
examples never enter the training order. The machine-readable training report
includes both losses plus the evaluated example and assistant-token counts.

Core hyperparameters (defaults are tuned for local Gemma4 SFT):

- `--training-steps` / `--steps` — number of optimizer steps (default `600`)
- `--batch-size` — training batch size (default `1`)
- `--learning-rate` / `--lr` — optimizer learning rate (default `0.0001`)
- `--rank` — LoRA rank (default `16`)
- `--alpha` — LoRA alpha (defaults to the rank)
- `--max-sequence-length` — maximum training sequence length (default `4096`)
- `--reasoning-effort` — Inkling renderer effort from `0` through `0.99`
  (default `0.9`; use the same value for inference)
- `--seed` — random seed (default `42`)
- `--target-modules` — comma-separated LoRA target suffixes (Gemma/Laguna
  default to q/k/v/o attention; Inkling also defaults to MLPs and `lm_head`)
- `--adapter-name` — adapter display name (default `local-assistant`)

This list is deliberately not exhaustive; run
`swift run mere.run text train-lora --help` for the full flag surface.

The optimizer projects only loss-masked target positions through the lm_head
(prompt and padding rows never contribute loss, so gradients are unchanged),
reads the loss back every 10 steps rather than per step, writes a
`<name>.partial.safetensors` checkpoint every 100 steps so a killed run can be
salvaged, and prints `[text-lora-train] step= loss= step_s= footprint_gb=`
diagnostics at each readback. `docs/configuration.md` documents the
`MERERUN_TEXT_LORA_TRAIN_*` and shared `MERERUN_LORA_TRAIN_*` environment
switches that tune or disable each behavior.

```bash
swift run mere.run text train-lora \
  --data ./pairs.seed.jsonl \
  --eval ./eval.prompts.jsonl \
  --output ./local-assistant.safetensors \
  --model text-chat-gemma4-12b-4bit \
  --visualize \
  --visualize-port 8787
```

Use the resulting adapter with native Gemma chat:

```bash
swift run mere.run text chat \
  --model text-chat-gemma4-12b-4bit \
  --lora ./local-assistant.safetensors \
  --prompt "What should this local assistant know?"
```

Laguna XS uses the same dataset and hyperparameter surface:

```bash
swift run mere.run text train-lora \
  --data ./pairs.seed.jsonl \
  --output ./laguna-xs-assistant.safetensors \
  --model text-chat-laguna-xs-2-1

swift run mere.run text chat \
  --model text-chat-laguna-xs-2-1 \
  --lora ./laguna-xs-assistant.safetensors \
  --prompt "What should this local assistant know?"
```

Laguna training defaults to q/k/v/o attention projections and writes
`mererun.laguna.text-lora` in its manifest. At inference time, applying an
adapter discards retained base-only QKV side layouts before generation so the
LoRA cannot be bypassed. Switching or removing an adapter reloads the base
model and is rejected while a continuous batch is active.

Inkling-Small uses the same assistant-only dataset and held-out evaluation
surface. Its native lane keeps the mixed checkpoint intact: routed experts
remain affine 2-bit/group-128, non-routed weights remain BF16, and only q/k/v/o
attention, gate/up/down MLP, and `lm_head` LoRA parameters are optimized. The
expert adapters use shared-outer factors: the hidden-dimension factor is shared
across experts while the expert-intermediate factor remains expert-specific.

```bash
swift run mere.run text train-lora \
  --data ./pairs.seed.jsonl \
  --eval ./eval.prompts.jsonl \
  --output ./inkling-assistant.safetensors \
  --model text-chat-inkling-small \
  --reasoning-effort 0.2

swift run mere.run text chat \
  --model text-chat-inkling-small \
  --lora ./inkling-assistant.safetensors \
  --reasoning-effort 0.2 \
  --prompt "What should this local assistant know?"
```

Inkling adapters write `mererun.inkling.text-lora` in their manifest. The
training graph uses a differentiable MLX mask path and treats discrete expert
selection indices as non-differentiable while retaining gradients through the
selected router scores and quantized expert computation.

Loss improvement is diagnostic, not a behavioral acceptance gate. The repository
includes a deterministic codebook task with 32 train examples and four unseen
paraphrases. The following proven recipe covers 12 complete shuffled epochs
using balanced batch-4 updates, then requires the adapter to answer all
held-out cases exactly while improving by at least three cases over the base
model:

```bash
swift run mere.run text train-lora \
  --model text-chat-inkling-small \
  --data Tests/MereRunCLITests/Fixtures/Inkling/receptivity-train.jsonl \
  --eval Tests/MereRunCLITests/Fixtures/Inkling/receptivity-eval.jsonl \
  --output .build/inkling-receptivity.safetensors \
  --reasoning-effort 0.2 --rank 4 --alpha 4 \
  --training-steps 96 --batch-size 4 --max-sequence-length 128

scripts/eval-inkling-receptivity.py \
  --model-root /path/to/Inkling-Small-MLX-Mixed-2bit \
  --adapter .build/inkling-receptivity.safetensors \
  --output .build/inkling-receptivity-report.json
```

The implementation validation on the pinned affine-q2/group-128 artifact used
that recipe on a 128 GB Apple Silicon machine. It measured a 110.2 GB peak,
held-out loss `4.4072 -> 0.0302`, and exact behavior `0/4 -> 4/4`. The gate
report records hashes for the CLI, model config, and adapter; those results
demonstrate this deterministic receptivity task, not broad downstream quality.

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/TextChatCommand.swift`
- `Sources/MereRunCLI/Commands/TextCodeCommand.swift`
- `Sources/MereRunCLI/Commands/TextEmbedCommand.swift`
- `Sources/MereRunCLI/Commands/TextAnonymizeCommand.swift`
- `Sources/MereRunCLI/Commands/TextTrainLoRACommand.swift`

### Chat families

- `Sources/MereRunCore/Q35/`
- `Sources/MereRunCore/Gemma4/`
- `Sources/MereRunCore/LFM2/`
- `Sources/MereRunCore/Psi/`
- `Sources/MereRunCore/MeBot/`

### Code generation

`mere.run text code` uses the vendored `llama.cpp` runtime via
`vendor/llama.xcframework` and the matching support code in `MereRunCore`. On
packaged Linux installs, the command uses the colocated `llama-cli` subprocess
so CUDA GGUF loads stay isolated from the MLX runtime in the Swift process.

### Embeddings

- `Sources/MereRunCore/Embeddings/`

### Anonymization

- `Sources/MereRunCore/PrivacyFilter/`

## Practical distinctions

### `mere.run text chat`

Use this for local assistant-style generation. It supports prompt/system
control, token limits, token streaming with `--stream`, and the chat-oriented
model families. Native MLX Gemma and Qwen-family models also support constrained
JSON objects with `--response-format json_object`.

### `mere.run text code`

Use this when you want a GGUF-backed coding path through the vendored
`llama.cpp` runtime. Linux packages run the same model family through the
bundled `llama-cli` subprocess when it is present.

### `mere.run text embed`

Use this for vector generation, semantic search prep, and other representation
tasks. It is not a generative command.

### `mere.run text anonymize`

Use this for local PII detection and redaction. It runs the OpenAI Privacy
Filter token-classification model through the native MLX runtime and can emit
plain redacted text or structured JSON spans.

### `mere.run text train-lora`

Use this to prepare and train Gemma-family, Laguna XS 2.1, or Inkling-Small text
LoRA adapters from reviewed chat SFT data. The Laguna lane is
`text-chat-laguna-xs-2-1`; Laguna S and DFlash are inference-only here. The
Inkling lane is `text-chat-inkling-small` and trains against its pinned native
mixed-precision MLX artifact.

## Reading the code

If you want to understand the text stack:

1. Start at the matching CLI command.
2. Follow model resolution through `MereRunModelManifest` and `ModelResolver`.
3. Continue to the family-specific runtime in `MereRunCore`.

For repository orientation, pair this page with
[CLI and runtime internals](../internals/cli-and-runtime.md).
