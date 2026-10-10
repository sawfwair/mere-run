# Native Whistle speech recognition

Whistle runs directly in Swift/MLX from the original
[Cactus-Compute archive](https://huggingface.co/Cactus-Compute/whistle).

```bash
mere.run model pull speech-asr-whistle
mere.run speech transcribe recording.wav --backend whistle
mere.run speech transcribe recording.wav --model speech-asr-whistle --language de
mere.run speech transcribe recording.wav --backend whistle --beam-size 3 --decoder-depth 4
mere.run speech transcribe recording.wav --backend whistle --keyword "Siobhan" --keyword "Mere Run"
mere.run speech listen --model speech-asr-whistle --keyword "Siobhan"
```

The managed download pins revision `ca5287601bef25af26dcf2e1b2bdc0843a7c19e5`
and downloads about 17 MB. A local model directory needs `config.json` and
`whistle.cact`. The archive supplies CQ2/CQ4 neural weights, FP16 scales and
small tensors, the exact tokenizer, and the audio filterbank. Native Metal
matrix operations read packed indices directly; Engram tables expand only the
requested rows. On Linux, portable MLX operations expand projection or requested
embedding rows on the CUDA device; they do not expand the full Engram table.
The runtime does not execute an upstream binary or use ONNX.

For the original FP32 reference path, also place
`checkpoints/whistle.safetensors` in that directory and select it explicitly:

```bash
mere.run speech transcribe recording.wav --backend whistle --model /path/to/whistle \
  --whistle-weights fp32 --beam-size 1 --no-timestamps
```

## Linux CUDA configuration

Packed Whistle uses portable MLX operations on Linux. Run it with CUDA graph
capture disabled because the pinned MLX backend's graph cache can exhaust its
entry limit during beam decoding:

```bash
MLX_USE_CUDA_GRAPHS=0 mere.run speech transcribe recording.wav --backend whistle
MLX_USE_CUDA_GRAPHS=0 mere.run speech listen --model speech-asr-whistle --jsonl
```

Set the same environment variable before initializing MLX when embedding the
runtime in another process. This is eager CUDA execution; it retains GPU tensor
operations. Linux performance and long-session behavior need broader qualification.

## Decoding controls

- English, German, French, Spanish, Italian, Dutch, and Polish transcription,
  with automatic language detection or a supported `--language` hint.
- Audio decoding and 16 kHz mono resampling through the shared AudioCodecs reader.
- Thirty-second inference windows with one-second overlap for longer files.
- Cached beam search: `--beam-size 1...8`, default 5. Size 1 is greedy.
- `--decoder-depth 2...8`, default 8, selects physical blocks by endpoint-preserving
  bisection. Depth 4 uses blocks 0, 3, 5, and 7; the encoder always uses all eight.
- Repeated `--keyword` phrases bias initial tokens by +2 and matching continuations
  by +5 logits. Up to 64 phrases, each up to 128 Unicode scalars.
- `--max-tokens` can lower the 318-text-token limit per window.
- Word timestamps use normalized cross-attention and dynamic time warping on an
  80 ms encoder grid. They are acoustic estimates. `--no-timestamps` skips alignment.
- Exact digital silence produces an empty transcript.
- Shared transcription operation records through `--run-dir` include the chosen
  Whistle controls and alignment results.

`--backend auto` keeps the existing Parakeet default. Naming the Whistle model
selects Whistle. Translation, unsupported language hints, and Core ML execution
are rejected. Whistle remains CLI-only.

## Live audio

```bash
mere.run speech transcribe recording.wav --backend whistle --stream
mere.run speech transcribe - --backend whistle \
  --stream --input-format pcm-s16le --sample-rate 16000 --jsonl
mere.run speech listen --model speech-asr-whistle --jsonl
```

Live sessions keep one model resident and use bounded VAD utterances, a bounded
input queue, partial revisions at `--stream-decode-ms` / `--decode-ms` cadence,
commit events, statistics, and one final event.
They re-decode the complete utterance at commit, including audio arriving after
its partial. Cancellation discards the active utterance. Live events carry
utterance times; word alignment is computed for file transcription only.
Whistle is a windowed speech model, so this is utterance streaming rather than
causal encoder inference.

## Qualification boundary

The native packed path uses the archive's CQ2/CQ4 weights with FP32 activations
and KV arithmetic. It does not reproduce the C++ engine's A8 activation/cache
quantization, beam heuristics, or noise rejection. The upstream footprint,
five-beam accuracy, and CPU latency measurements describe that engine; they are
not measurements of this runtime's total memory, recognition accuracy, or latency.

Numerical tests compare encoder cross-K/V and four cached decoder steps against
independent graphs for both original FP32 and independently dequantized CQ weights.
Asset-free tests exercise CQ2/CQ4 Metal matrix products, sparse row gathers,
decoder layer selection, keyword biasing, and the frontend against a scalar DFT.
Multilingual accuracy, long-file overlap quality, timestamp accuracy, and real
microphone capture need a broader corpus and device qualification before claiming
parity with the upstream engine.

## Numerical checks

An optional complete-checkpoint comparison accepts `MERERUN_WHISTLE_PARITY_DIR`,
containing `whistle.safetensors`, `whistle.cact`, `whistle-reference.json`, and
`whistle-cq-reference.json`. Generate the JSON references with
`scripts/fixtures/export-whistle-reference.py`, using `--cact` for the CQ oracle.
The fixture's [README](https://github.com/sawfwair/mere-run/blob/main/Tests/SpeechRuntimeTests/Fixtures/Whistle/README.md)
documents asset hashes and regeneration. Graphs are validation inputs only.

```bash
MERERUN_WHISTLE_PARITY_DIR=/path/to/parity swift test --filter Whistle
./scripts/check.sh
```

See the [upstream architecture](https://cactuscompute.com/blog/whistle) and
[container notes](https://cactuscompute.com/blog/cact) for model conventions.

## Local qualification

On 2026-10-10 the pinned archive was checked with the public space's 4.259-second
English example. Packed greedy and five-beam decoding both produced “Please turn
off the kitchen lights and set an alarm for 7.30 tomorrow morning.” Five-beam
word alignment returned 14 ordered word spans within the audio duration. The
explicit FP32 greedy path produced the same words, spelling the time “730”.

Paced PCM stdin produced a partial, a complete commit, statistics, and exactly one
EOF final event. The four-layer greedy path ran but misrecognized “set an” as
“said in” on this example, so reduced depth is a speed/accuracy tradeoff, not a
qualified replacement for eight layers. A fresh managed install validated at
16.9 MB without downloading the FP32 checkpoint. These are bounded smoke results,
not corpus accuracy or real-microphone qualification.

Optimized release builds also passed five-beam word alignment and resident PCM
streaming from the compact install, with the same complete transcript.
