# YuE2

This guide is for Studio and CLI users who generate songs with `music-yue2`.
YuE2 uses a musical style and lyrics to plan an ABC score, generate music tokens,
and decode 48 kHz stereo audio with native Swift/MLX.

This integration is experimental. The released checkpoints passed bounded native
generation checks, including a 36.08-second vocal track that ended naturally.
Broader listening quality and long-song performance remain unqualified. Numerical tests
also compare small random-weight models against the pinned upstream implementation.

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
unchanged regions of an existing waveform. Audio-to-score transcription with
SheetSage2 is outside this integration. ACE-Step editing, adapters, MiniMax
composition, and music server controls do not apply to YuE2.

The native runtime loads BF16 transformer weights, releases the transformer,
and then decodes with the standard FP32 VAE. The experimental memory estimates
are 24 GB minimum and 32 GB recommended; long-song peak memory is unmeasured.
Legacy benchmark decoding and runtime quantization are not supported.

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

Both weight licenses require attribution and noncommercial use. Model pulls
require explicit acceptance; generation never downloads these weights implicitly.

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
