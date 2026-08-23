# MiniMax Music 3 Generate

## Purpose

Generate a complete lyric-conditioned song locally with the pinned native
Swift/MLX MiniMax Music 3 runtime.

## Install

Review the pinned MiniMax-Music3 Community License, then pull the selective
Diffusers-format runtime snapshot:

```bash
mere.run model pull music-minimax-music3 --accept-model-license
mere.run model info music-minimax-music3
```

The managed pull intentionally excludes the duplicate original SGLang weights,
reference media, figures, and Python example. It contains every file consumed
by the native runtime and cannot be used as an SGLang checkpoint directory.

## Generation contract

MiniMax Music 3 consumes only these model inputs:

- positional caption: genre, emotional arc, vocals, instrumentation,
  arrangement, and production profile
- `--compose`, `--lyrics`, `--lyrics-file`, `--lrc-file`, or `--instrumental`
- `--lyrics-preflight off|warn|strict`: duration and section checks; defaults
  to `warn`
- `--duration`: upper-bound seconds; defaults to 60
- `--minimum-duration`: decoded-audio duration floor. The runtime rounds up to
  the first 25 Hz frame whose whole 512-sample vocoder hops meet the request.
- `--min-frames`: exact acoustic-frame floor; 1 through 9000
- `--max-frames`: exact 25 Hz acoustic-frame upper bound; 1 through 9000
- `--sampling-tier quality|fast|draft`: named flow schedules with 30,
  20, or 16 steps; defaults to `quality`
- `--steps`: exact flow steps per chunk; overrides `--sampling-tier`
- `--profile-output`: write synchronized stage timings as JSON; intended for
  benchmarking because the extra evaluations can affect wall time
- `--seed`: deterministic MLX seed; defaults to 0
- `--guidance-scale`: flow classifier-free guidance; defaults to 1.7
- `--memory-mode staged|resident`: staged loads and releases the autoregressive,
  flow, and vocoder stages separately; resident keeps every component loaded
- `--performance-mode reference|optimized|q8-lm|q4-lm|q8|q4`: exact upstream
  graph, parity-safe BF16 acceleration (default), split LM-only affine
  quantization with BF16 depth, or legacy whole-autoregressive quantization
- `--sample-rate 44100|32000`: native Diffusers output or SGLang-compatible WAV
- `--flow-solver euler|ab2`: parity-default Euler or opt-in second-order AB2
- `--ar-cfg-frames`: opt-in count of autoregressive frames that retain CFG
- `--flow-cfg-end`: opt-in normalized cutoff after which flow runs only the
  conditional row

When both `--duration` and `--max-frames` are provided, they must describe the
same limit. The checkpoint hard cap is 9000 frames (360 seconds), while the
official supported-quality claim is songs up to five minutes.

`optimized` keeps BF16 weights while reducing launch and memory traffic with a
compact reachable-token head, fused global-language-model projections, a
fixed-capacity global KV cache, and batched flow guidance. The short residual
depth prefix is deliberately recomputed to preserve seeded lyric trajectories.
`q8-lm` additionally quantizes the global language model with group-64 affine
weights while retaining the residual-depth decoder in BF16; `q4-lm` applies
the same split at four bits. These are the preferred quantized experiments
because depth codebooks directly shape vocal detail. Legacy `q8` and `q4`
continue to quantize both autoregressive components for compatible maximum
compression. Quantization changes the sampled composition, so use `reference`
or `optimized` for upstream-parity investigations.

The `quality`, `fast`, and `draft` sampling tiers use the same model and fixed
Euler schedule with 30, 20, and 16 evaluations respectively. Use `--steps`
when you need an exact custom count; it takes precedence over the named tier.
Euler and full classifier-free guidance remain the default parity path.
`--flow-solver ab2` uses Euler for the first update and second-order
Adams-Bashforth for later updates. `--ar-cfg-frames 50` and
`--flow-cfg-end 0.4` switch their later work to conditional-only execution.
Treat these as reproducible full-song A/B controls, not speed or quality
presets.

Incompatible ACE-Step cover, editing, LM-planner, adapter, VAE,
candidate-ranking, stem, and DAW settings fail explicitly for this model.
Magenta RT2 settings also fail instead of being silently ignored.

## Upstream parameter map

| Upstream Diffusers input | mere.run surface |
| --- | --- |
| `prompt` | positional caption |
| `lyrics` | `--compose`, `--lyrics`, `--lyrics-file`, `--lrc-file`, or `--instrumental` |
| `audio_duration` | `--duration` |
| `minimum_audio_duration` | `--minimum-duration` |
| `min_new_tokens` | `--min-frames` |
| `max_new_tokens` | `--max-frames` |
| `generator` | `--seed` |
| `num_inference_steps` | `--steps` |
| `output_type=np|pt` | `--export-format float32` with mastering disabled as shown below |

The released autoregressive CFG (`1.5`) and top-k (`50`) remain fixed while CFG
is active. Flow guidance defaults to the released `1.7` and is available as
`--guidance-scale`. Guidance cutoffs are explicit experimental execution
controls and never change those scales. The speech route
maps upstream `max_new_tokens` to the same frame limit exposed by
`--max-frames`; native `minimum_audio_duration` and `min_new_tokens` map to the
same duration floor exposed by `--minimum-duration` and `--min-frames`. It
accepts only `response_format=wav`, and requires
`stream=false`. Both upstream paths are single-sample, so mere.run does not
invent a batch parameter.

## Lyrics

Put structural tags such as `[Verse]`, `[Chorus]`, `[Bridge]`, and
`[Instrumental]` on their own lines. Text placed after a leading tag on the
same line is discarded by the upstream checkpoint contract.

```bash
mere.run music generate \
  "Genre: acoustic pop. BPM: 96. Warm female lead, intimate verses, wide final chorus." \
  --model music-minimax-music3 \
  --lyrics-file ./lyrics.txt \
  --duration 60 \
  --minimum-duration 60 \
  --performance-mode q8-lm \
  --steps 30 \
  --seed 7 \
  --output ./song.wav
```

Use `--instrumental` to pass the upstream `[Instrumental]` marker without a
lyrics file.

`--duration` is an upper bound, so sparse lyrics can let the autoregressive
model emit EOS well before it. The default lyric preflight prints warnings for
underfilled long-form lyrics, missing structure, and unsupported or inline
tags. `--lyrics-preflight strict` fails before checkpoint loading when any
issue remains. It does not add `--minimum-duration` or otherwise force EOS
masking on the user's behalf.

## Local composer

`--compose` turns the positional caption into a natural-language song brief.
A local native Gemma4 or Qwen-family chat model runs two constrained-JSON
passes: a BPM/meter/section blueprint whose bars fill the requested timeline,
then the final title, tags, lyrics, and `Global Metadata`, `Vocal Details`, and
`Arrangement` fields. The typed contract normalizes the bar budget, requires a
supported ordered section sequence ending in an outro, checks
per-section lyric budgets, and keeps production directions out of lyric lines.
Each phase gets at most one validation-guided repair attempt while the writer
model remains loaded.

```bash
mere.run music generate \
  "slow-burn dream pop about leaving a familiar city and finding home" \
  --model music-minimax-music3 \
  --compose \
  --duration 180 \
  --lyrics-preflight strict \
  --performance-mode q8-lm \
  --sampling-tier fast \
  --output ./composed-song.wav
```

If tagged `--lyrics` or `--lyrics-file` is also present, that text is authoritative:
the composer plans its existing section order and returns it unchanged while
writing the caption. `--instrumental` creates a section-only timeline. Use
`--composer-model`, `--composer-model-root`, and
`--require-composer-installed` to control the writer model. The writer unloads
before MiniMax loads. `<output>.composition.json` records the request,
blueprint, finished inputs, writer model, and preflight; schema 7 recipe JSON
embeds the same provenance.

## Upstream structured-caption companion

MiniMax publishes a separate `music-caption-rewriter` agent skill containing
its structured-caption workflow and static template library. It is not
vendored because the companion repository publishes no reusable license.
Install it directly from the official source when you want it:

```bash
npx skills add MiniMax-AI/MiniMax-Music3 --skill music-caption-rewriter
```

The integrated `--compose` workflow is a typed local mere.run implementation;
it does not vendor that upstream skill. If you use the upstream companion
instead, pass its `Global Metadata`, `Vocal Details`, and `Arrangement` text as
the positional caption while keeping the original lyrics separate.

## Native and reference-server output

The native vocoder emits 44.1 kHz stereo. `--sample-rate 32000` adds the
reference-server compatibility resample. WAV export defaults remain mere.run's
production profile: PCM24, peak normalization to -1 dBFS, short boundary fades,
and deterministic dither.

For an unmastered Diffusers-style float waveform, use:

```bash
--export-format float32 --normalize none --fade-in-ms 0 --fade-out-ms 0 --no-dither
```

## Resident speech API

Start the SGLang-compatible speech route with all weights warm:

```bash
mere.run music serve \
  --model music-minimax-music3 \
  --memory-mode resident \
  --performance-mode q8-lm \
  --port 8081
```

Then send lyrics as `input` and the caption as `instructions`:

```bash
curl http://127.0.0.1:8081/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMaxAI/MiniMax-Music3",
    "input": "[Verse]\nMorning light through the pine",
    "instructions": "Warm acoustic pop, intimate female vocal, fingerpicked guitar.",
    "response_format": "wav",
    "seed": 7,
    "max_new_tokens": 750,
    "min_new_tokens": 750,
    "sampling_tier": "fast",
    "flow_solver": "ab2",
    "autoregressive_guidance_frames": 50,
    "flow_guidance_end": 0.4,
    "lyric_preflight": "strict",
    "stream": false
  }' \
  --output song.wav
```

The speech route defaults to 32 kHz PCM16 stereo. Add `"sample_rate": 44100`
for the native vocoder rate. Streaming and batched generation are not offered
because the released upstream pipeline is single-sample and non-streaming.

## Provenance

- Checkpoint: `MiniMaxAI/MiniMax-Music3` at the immutable revision recorded by
  `mere.run model info music-minimax-music3`
- Runtime reference: Hugging Face Diffusers PR #14456
- Caption companion: `MiniMax-AI/MiniMax-Music3`, commit
  `91410fb657c007ae57c60df8240f5ece5be089c7`
