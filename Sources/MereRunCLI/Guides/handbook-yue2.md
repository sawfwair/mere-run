# YuE2

This guide is for Studio and CLI users who generate songs with `music-yue2`.
YuE2 uses a musical style and lyrics to plan an ABC score, generate music tokens,
and decode 48 kHz stereo audio with native Swift/MLX.
The companion models are `music-sheetsage2` for transcription and
`music-yue2-vae-legacy` for decoding with the upstream benchmark VAE.

This integration is experimental. The released checkpoints passed bounded native
generation checks, including a 36.08-second vocal track that ended naturally.
The November 7 M5 Max qualification also covers released-weight numerical
comparisons, reusable stages, a six-minute stress run, and a clean instrumental
excerpt accepted by the owner. One style-only instrumental planning attempt
initially exhausted its score budget; matching upstream's planner-only section
tags made the same case finish within the default limit. Broader listening quality and
device coverage remain unqualified. See `docs/runtime/yue2-qualification.md`
in the repository for the measured scope and limits.

## Example to adapt

After reviewing the CC BY-NC 4.0 terms for both model components, install them:

```bash
mere.run model pull music-yue2 --accept-model-license
```

Generate a song from an original lyric file:

```bash
mere.run music generate "English piano pop, warm alto, soft drums" \
  --model music-yue2 \
  --lyrics-file ./lyrics.txt \
  --duration 60 \
  --seed 831001 \
  --output ./song.wav
```

The command saves `song.wav`, `song.abc`, and `song.recipe.json`. The recipe
records effective controls, score IDs, music codes, token-limit truncation,
export processing, and the output hash. These commands illustrate the interface;
the checkpoint verification below records the tested settings.

## Controls and variants

| Control | Behavior |
| --- | --- |
| `--score-mode full` | Plan a chord-annotated ABC score; the default. |
| `--score-mode melody` | Plan a melody-only score without chords. |
| `--score-mode off` | Generate directly from style and lyrics. |
| `--abc-file ./score.abc` | Supply a score in full or melody mode; skip score generation. |
| `--abc-output ./score.abc` | Choose the saved score path. |
| `--abc-max-tokens 4096` | Set the generated score token budget. |
| `--duration 60` | Limit generation to 1,500 frames at 25 Hz. It is an upper bound. |
| `--min-frames` / `--max-frames` | Set the music frame floor and ceiling, up to 9,000. |
| `--minimum-duration` | Set a decoded duration floor, including the decoder's 64-sample end crop. |
| `--steps 32` | Run 32 midpoint integration steps, with two velocity evaluations per step. |
| `--guidance-scale` | Set semantic CFG: 1 with a score, or 1.01 in off mode by default. |
| `--semantic-temperature` | Set music sampling temperature; 1 by default, 0 for greedy selection. |
| `--semantic-top-p` / `--semantic-top-k` | Set music token filters; 0.95 and 100 by default. |
| `--semantic-repetition-penalty` | Set a frequency penalty over the last 50 music tokens; 1.2 by default. |

The default music budget is 9,000 tokens, approximately six minutes. The score
sampler defaults to temperature 0.7, top-p 0.9, top-k 30, and a 1.005 frequency
penalty over 100 tokens. A prompt and its generation budget must fit the 24,576-token
context. Overlong inputs produce an error; inputs are not silently truncated.

The decoder emits `1920 × frames - 64` samples. A 50-frame result is slightly
shorter than two seconds. A two-second minimum therefore requires 51 frames.
An end token can finish a song before its maximum duration. Inspect the recipe
for truncation when generation reaches the limit.

To render an edited or independently prepared score, run the following command:

```bash
mere.run music generate "English jazz, brushed drums, upright bass" \
  --model music-yue2 \
  --lyrics-file ./lyrics.txt \
  --score-mode melody \
  --abc-file ./melody.abc \
  --output ./variation.wav
```

Score conditioning generates a complete recording. It does not preserve
unchanged regions of an existing waveform. ACE-Step editing, adapters, and MiniMax
composition controls do not apply to YuE2.

`--instrumental` requires empty lyrics and full or melody score mode. When YuE2
plans a new score, the planner receives Intro, Verse, Chorus, and Outro section
tags, matching the upstream instrumental helper. The rendering lyrics stay empty.
It transfers vocal notes into the instrumental voice before music generation, preserving their
pitch, onset, and duration. Vocal notes take priority over overlapping
accompaniment. Listen for vocal leakage; symbolic conversion does not guarantee
voice-free audio. To convert an existing score without loading a model:

```bash
mere.run music yue2 instrumentalize ./song.abc --output ./instrumental.abc
```

Add `--overlap error` to reject overlapping accompaniment, or `--drop-chords`
for a melody-only cover.

`--yue2-memory-mode staged` releases the transformer before VAE decoding.
`resident` retains weights between requests on one operation. `low-memory`
loads the AR and NAR layer branches in separate phases, retaining each chunk's
conditioning cache while switching. Extra checkpoint reads can reduce speed.
See `docs/runtime/yue2-qualification.md` for six-minute peak memory and performance
measurements. Smaller-device memory requirements remain unqualified.

## Reuse generation stages

Add `--stages-output ./song-stages` to `music generate` to retain separate
`score/`, `semantic/`, and `acoustic/` stages. Each contains `stage.json`;
score-bearing stages include `score.abc`, and the acoustic stage includes
FP32 `latent.npy` with shape `[frames,64]`.

A minimal native request is:

```json
{"style":"gentle piano instrumental","instrumental":true,"seed":42}
```

Save this JSON as `request.json`, then run each stage separately:

```bash
mere.run music yue2 validate request.json
mere.run music yue2 validate request.json --tokenizer ./model/qwen.tiktoken
mere.run music yue2 plan --request request.json --output ./score-stage
mere.run music yue2 semantic ./score-stage --output ./semantic-stage
mere.run music yue2 synthesize ./semantic-stage --output ./acoustic-stage
mere.run music yue2 inspect ./acoustic-stage
mere.run music yue2 decode ./acoustic-stage --output ./song.wav
```

Requests use native plan field names: `style`, `lyrics`, `planning`, `abc`,
`seed`, `guidanceScale`, `scoreSampling`, `semanticSampling`, `steps`, and
`instrumental`. Omitted top-level fields receive generation defaults. A supplied
sampling object must include every sampling field; the `validate` command prints
the complete effective request to use as a starting point. `planning` is
`full`, `melody`, or `off`. Stage continuation uses the saved effective request;
to change a composition, create another request with its edited `abc` text.

Stage receipts record exact model, tokenizer, and configuration hashes.
The optional tokenizer preflight reports exact prompt length, the next generation
budget, and acoustic chunk sizes when a score is supplied or planning is off.
It loads no model weights. A generated score requires another context check
after planning, so its acoustic chunk sizes are not predicted.
Inspection checks artifact hashes without loading models. Resuming score or
semantic stages requires their original model files and exact tokenized score.
Do not modify receipts to change a generation. Hashes check consistency with
the receipt; they do not independently authenticate provenance or assess quality.
Outputs must be fresh paths. Cancellation or failure retains completed stages.

Decoder-only runs load no transformer or tokenizer. `decode` also accepts
an upstream NumPy latent file in `[T,64]` or `[1,64,T]` format. Use `--vae`
with a local decoder directory, `--core-frames` to set tile size, or `--full`
for whole-sequence decoding. Decode writes float32 WAV without normalization
or fades. For benchmark-decoder comparisons:

```bash
mere.run model pull music-yue2-vae-legacy --accept-model-license
mere.run music yue2 decode ./acoustic-stage \
  --vae music-yue2-vae-legacy --output ./benchmark-decoder.wav
```

Benchmark candidate selection remains separate from a generation call. A
single native run is not the upstream best-of-two or best-of-eight protocol.

## Transcribe and cover recordings

Install the separate SheetSage2 adapter and its pinned MERT2 encoder:

```bash
mere.run model pull music-sheetsage2 --accept-model-license
mere.run music yue2 transcribe ./source.wav --melody-only --output ./transcription
mere.run music yue2 cover ./source.wav --request ./cover-request.json --output ./cover
```

The cover request supplies the target `style` and optional new `lyrics`.
A cover transcribes a chord-free melody by default; `--full-score` retains
source harmony. Set `instrumental` in the request to transfer the transcribed
vocal melody into the instrumental voice. Transcription completes before YuE2
loads, so both models do not stay resident together.

Transcription uses mono 24 kHz audio, 300-second windows, 200-second overlap,
and 100-second right-hand lookahead. `--overlap` and `--lookahead` change these
last two settings. Raw event tokens, timing, window records, and truncation
status are retained in `transcription.json`. The output also includes raw
melody MIDI and LAB files for melody, key, chords, and structure. Unsupported notation leaves this
raw record and reports a score-export error. Use `--allow-notation-failure`
to accept the raw JSON, MIDI, and LAB analysis as the result. The native score exporter uses
YuE2's two-voice dialect and requires notes on its 1/32 grid and meter changes
at measure boundaries.

SheetSage2 is experimental. Independent upstream CPU references cover both
small fixtures and released-weight arithmetic. Production-window transcription
matched all 190 upstream tokens on a 32-note synthetic melody; onset-and-pitch
F1 was 0.985, with one extra note. That bounded case does not establish general
transcription accuracy. See the repository qualification report for longer
recordings, remaining limits, and exact test conditions.

## Offline export, batches, and serving

```bash
mere.run music yue2 export --output ./offline-yue2
mere.run music yue2 batch ./requests.jsonl --output ./batch-runs
mere.run music yue2 serve --output ./server-runs --port 8082
```

Offline export copies local weights, configuration, tokenizer, and licenses
with a hash manifest. It does not download files or copy executable model code.
Point `--model` at that directory on the destination machine.

Each batch line contains `id` and `request`, for example:
`{"id":"piano-1","request":{"style":"piano","instrumental":true}}`.
`--validate-only` checks the requests without loading models. Invalid or
duplicate IDs and existing outputs fail that item; other items continue.
Stdout contains one JSON outcome per input line.

The local job API accepts native request JSON at `POST /v1/jobs`, returns a
job ID, and exposes `GET /v1/jobs/<id>`, `POST /v1/jobs/<id>/cancel`, and
`GET /v1/jobs/<id>/audio`, `/score`, or `/latents`. Jobs execute serially on
a shared operation. `--max-pending` bounds admitted work. The default memory
mode is `resident`. Non-loopback binds require `--api-key` or `MERERUN_API_KEY`.
Job status is process-local; completed stage files remain inspectable after
restart. Manage retained files in the selected output directory.

The native runtime uses MLX attention and execution. Upstream PyTorch compilation,
CUDA/XPU graphs, vLLM batching, and FP8 AR controls are backend-specific and are
not native MLX settings. NVIDIA throughput claims do not qualify this port.

Use a fixed seed to compare native runs with identical settings. The native
MLX random generator differs from upstream PyTorch, so the same seed does not
reproduce an upstream recording. Export normalization and fades follow the
shared music command. To inspect raw decoder samples, set `--export-format
float32 --normalize none --fade-in-ms 0 --fade-out-ms 0`.

## Sources and validation

- [YuE2 source](https://github.com/multimodal-art-projection/YuE/tree/0edaf2f4053ef4731334b8329834b107977f9637): Apache 2.0,
  with MIT-licensed Oobleck and SnakeBeta components.
- [YuE2-3B weights](https://huggingface.co/m-a-p/YuE2-3B/tree/29b3558dd46954a0cd9021dc76d5c91864a0f1c7): CC BY-NC 4.0.
- [Standard VAE weights](https://huggingface.co/m-a-p/YuE2-Vae/tree/9a94e1d0ea9f8087e98f77fa88df4a4068104d2a): CC BY-NC 4.0.

The pinned weight files retain their CC BY-NC 4.0 licenses. Upstream also
publishes individual creator permission, including output monetization, in its
[model license](https://github.com/multimodal-art-projection/YuE/blob/1116bea76032bee5668e5f097294d082fa0b84f0/MODEL_LICENSE).
Company commercial use requires separate authorization. SheetSage2 and MERT2
retain their own weight terms. Model pulls require explicit acceptance;
generation never downloads these weights implicitly.

The native tests cover prompts, BPE behavior, vocabulary filtering, BF16 and
FP32 transformer outputs, cached decoding, midpoint integration, and decoder
tile boundaries. An optional installed-checkpoint test compares full and tiled
decoding with the released VAE weights.

### Checkpoint verification

On September 15, 2026, the debug CLI completed five runs on an Apple M4 Max
with 128 GB of memory. The downloaded transformer and VAE SHA-256 hashes matched
the pinned repository metadata. Every run used 32 midpoint steps and seed 831001.

| Run | Audio duration | Wall time | Result |
| --- | --- | --- | --- |
| Direct generation without a score | 8.00 s | 35.05 s | Reached the requested frame limit. |
| Repeat of direct generation | 8.00 s | 26.09 s | Byte-identical WAV. |
| Full score planning and vocal generation | 30.00 s | 79.37 s | Completed a 326-token score; reached the music frame limit. |
| Supplied score with an eight-second limit | 8.00 s | 21.05 s | Preserved score IDs and the original semantic token prefix. |
| Supplied score with a 90-second limit | 36.08 s | 73.31 s | Emitted the music end token; no score or music truncation. |

All outputs were finite, non-silent 48 kHz stereo audio with the expected sample
count, successful receipts, and matching recipe hashes. The final run recorded
a 19.99 GB process peak memory footprint and 14.68 GB maximum resident size.
These are single-run observations from a debug build, not throughput benchmarks
or proof of the minimum memory estimate. The 36.08-second sample passed an owner
listening review. Broader listening quality, other languages, long-song generation,
and cancellation during generation remain unqualified.
