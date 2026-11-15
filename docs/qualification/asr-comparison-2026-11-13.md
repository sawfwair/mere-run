# Native ASR comparison — 2026-11-13

All five supported ASR families ran through the same release Swift executable
on an Apple M5 Max with 128 GiB unified memory. The table compares current native
pipelines, their default checkpoints, and selected decoder/provider controls.
All models transcribed the short reference without word errors. Long-recording
scores are from one 129-word human-speech sample; they do not establish general
ASR accuracy or language coverage.

## Resident latency

| Model / execution | 3.52 s recording | 46.57 s recording | Long audio / inference time | Long word errors |
| --- | ---: | ---: | ---: | ---: |
| Parakeet 0.6B v3 · MLX BF16 | 38.0 ms | 176.6 ms | 263.7× | 4/129 (3.10%) |
| Parakeet 0.6B v3 · Core ML | 42.2 ms | 200.6 ms | 232.2× | 4/129 (3.10%) |
| Qwen3-ASR 1.7B · MLX 8-bit | 99.7 ms | 785.8 ms | 59.3× | 2/129 (1.55%) |
| SOV · BF16 + DFlash | 299.4 ms | 1.364 s | 34.1× | 2/129 (1.55%) |
| SOV · custom int8 FFN + DFlash | 285.0 ms | 1.369 s | 34.0× | 2/129 (1.55%) |
| SOV · palettized Core ML + DFlash | 460.7 ms | 1.742 s | 26.7× | 2/129 (1.55%) |
| SOV · BF16 · target only | 472.6 ms | 3.091 s | 15.1× | 2/129 (1.55%) |
| Nemotron 3.5 Transcribe 30B-A4B · expert INT4 | 370.7 ms | 3.640 s | 12.8× | 2/129 (1.55%) |
| Whistle · packed · greedy | 405.9 ms | 4.400 s | 10.6× | 4/129 (3.10%) |
| Whistle · packed · beam 5 | 1.742 s | 21.431 s | 2.2× | 4/129 (3.10%) |

Each entry is a median of five measured runs after two warmups on each recording.
One fresh process loads each variant once and keeps it resident across both
recordings. The controlled session holds temporary display and system idle-sleep
assertions for each process with `caffeinate -di`. All inference processes run
serially, outside Instruments and without concurrent builds. File decoding and
checkpoint loading are excluded from resident latency; feature extraction,
prefill, generation, alignment when applicable, cleanup, and stream completion
are included. The output budget is 448; Whistle retains its native per-window
318-token ceiling. Whistle uses full decoder depth and text-only output.
Parakeet's intrinsic alignment remains included.

These are complete backend paths with their own windowing policies. Parakeet
MLX processes the long recording as one window, whereas its static Core ML
encoder uses five overlapping/padded windows. SOV uses two 30-second windows.
Core ML decoder speed and encoder speed must therefore be interpreted alongside
window counts, rather than attributed entirely to hardware.

## Loading and memory

| Model / execution | Model preparation | First inference after preparation | Peak process RSS |
| --- | ---: | ---: | ---: |
| Parakeet 0.6B v3 · MLX BF16 | 0.258 s | 0.052 s | 0.85 GiB |
| Parakeet 0.6B v3 · Core ML | 0.691 s | 0.049 s | 0.17 GiB |
| Qwen3-ASR 1.7B · MLX 8-bit | 0.697 s | 0.278 s | 2.50 GiB |
| SOV · BF16 + DFlash | 16.074 s | 21.834 s | 70.82 GiB |
| SOV · custom int8 FFN + DFlash | 15.326 s | 14.833 s | 73.05 GiB |
| SOV · palettized Core ML + DFlash | 15.602 s | 5.238 s | 73.62 GiB |
| SOV · BF16 · target only | 15.354 s | 8.400 s | 71.00 GiB |
| Nemotron 3.5 Transcribe 30B-A4B · expert INT4 | 0.560 s | 2.422 s | 21.10 GiB |
| Whistle · packed · greedy | 0.047 s | 0.425 s | 0.14 GiB |
| Whistle · packed · beam 5 | 0.047 s | 1.764 s | 0.14 GiB |

These are observed fresh-process timings without clearing filesystem or shader
caches. The first input is the short clip. Preparation and first inference are
separate: their sum plus audio decoding approximates the first request.
Process RSS includes loading and warmup and comes from `getrusage`; it does not
measure every driver/ANE-service allocation. Per-run MLX active-memory peaks,
which exclude Core ML allocations, are preserved separately in the JSON.
Core ML process RSS was about 1.3 GiB during the initial compilation/load and
0.17 GiB with the warmed service cache; RSS alone cannot establish total ANE
model memory.

## Variability and checks

| Model / execution | Short measured range | Long measured range |
| --- | ---: | ---: |
| Parakeet 0.6B v3 · MLX BF16 | 37.9 ms–38.5 ms | 176.4 ms–177.2 ms |
| Parakeet 0.6B v3 · Core ML | 41.9 ms–42.7 ms | 197.1 ms–200.7 ms |
| Qwen3-ASR 1.7B · MLX 8-bit | 99.6 ms–100.0 ms | 784.6 ms–790.6 ms |
| SOV · BF16 + DFlash | 293.2 ms–299.7 ms | 1.362 s–1.384 s |
| SOV · custom int8 FFN + DFlash | 283.7 ms–285.5 ms | 1.363 s–1.392 s |
| SOV · palettized Core ML + DFlash | 460.0 ms–461.5 ms | 1.727 s–1.766 s |
| SOV · BF16 · target only | 472.0 ms–473.6 ms | 2.999 s–3.112 s |
| Nemotron 3.5 Transcribe 30B-A4B · expert INT4 | 369.0 ms–375.6 ms | 3.619 s–3.716 s |
| Whistle · packed · greedy | 404.6 ms–407.3 ms | 4.396 s–4.411 s |
| Whistle · packed · beam 5 | 1.737 s–1.767 s | 21.377 s–21.469 s |

The raw receipt also retains the initial two-session comparison, which reversed
model and recording order and recorded ten measured runs per variant/clip.
That batch showed substantial SOV timing swings, including slower custom-int8
runs. A separate held-awake session returned timings closer to the earlier
standalone SOV measurements. The primary table uses the same explicit awake
condition for every model. This association does not isolate idle power as the
sole cause of the earlier variance, and small SOV end-to-end differences should
not be treated as a guaranteed speedup. No observations from the initial batch
were discarded.

Scoring lowercases text, ignores punctuation, normalizes curly apostrophes, and
preserves contractions and spelling differences. Both recordings, references,
checksums, every measured transcript, and all timing samples are in the JSON.
All SOV measured token sequences were checked against the earlier frozen
13-/138-token references; see `sovFrozenTokenParity` in the receipt.

The Parakeet providers both scored 0/11 short and 4/129 long word errors, but
their long transcripts differ in proper-name spelling. Different windowing and
FP16/BF16 arithmetic are part of these provider paths; bit-exact cross-provider
transcript parity is not established by equal WER.

## Xcode compatibility and ANE execution

The converter uses the selected installed Xcode and records its actual version.
The loader validates the artifact format, source pin, conversion dependencies,
and complete file-hash closure, with compiler version retained as provenance.
It accepts compatible artifacts without matching a hard-coded Xcode release.
Legacy, 27.2, and future compiler metadata have regression coverage; blank
provenance remains invalid.

The new Parakeet artifact was built with Xcode 27.2 / build 27B52. Encoder trace
and decoder-state checks had zero maximum absolute error before FP16 conversion.
Every nonconstant encoder/decoder operation prefers ANE. A separate Instruments
trace recorded two encoder and forty decoder ANE prediction intervals, confirming
hardware execution. Profiling was excluded from benchmark timing.

## Reproduce

```bash
swift build -c release --product mere.run
python3 scripts/validation/benchmark_asr.py --dry-run
python3 scripts/validation/benchmark_asr.py --rounds 1 --keep-awake \
  --output .build/qualification/asr-comparison/awake-recheck
```

The script requires local checkpoints at the paths shown by its plan and never
downloads models while scoring. `model benchmark asr` accepts arbitrary installed
roots for individual models. See [benchmarking](../benchmarking.md#native-asr-timing)
for options. The source, binary, checkpoint revisions, command receipts, compiler
verification, and full samples are recorded in the [JSON report](./asr-comparison-2026-11-13.json).
The independent [ANE hardware event export](./asr-comparison-2026-11-13-ane-events.xml) records
execution. The release build, 15 focused Swift tests, eight converter tests,
and strict lint passed. The full repository gate remains deferred as requested.
