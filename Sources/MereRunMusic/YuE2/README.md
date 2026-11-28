# YuE2

This directory owns native Swift/MLX inference for `music-yue2`. The implementation
follows YuE2 source revision `1116bea76032bee5668e5f097294d082fa0b84f0`:

1. `YuE2Tokenizer` reads the frozen Qwen byte BPE and normalizes text to NFC.
2. `YuE2GenerationPlan` owns validated requests and exact prompt markers.
3. `YuE2Model` runs separate autoregressive and acoustic transformer branches.
4. `YuE2Sampler` masks text and codec vocabularies, applies windowed frequency
   penalties, and retains the score IDs in the semantic CFG negative prefix.
5. `YuE2Acoustic` caches causal prefix keys once per original context chunk and
   integrates acoustic latents with the upstream midpoint schedule. Both zero
   boundary frames participate in bidirectional attention.
6. `YuE2Decoder` loads the standard FP32 Oobleck decoder, folds weight
   normalization, and crops exact tile cores using an audited receptive field.
7. `YuE2GenerationOperation` owns stream lifetime, cancellation, stage reuse,
   phase loading, resident weights, finite-output checks, and waveform conversion.
8. `YuE2Score` parses the native two-voice dialect and moves vocal notes to the
   instrumental voice, checking note identity after serialization.
9. `YuE2SavedStage` stores immutable model identities, exact requests, tokens,
   and NumPy acoustic latents. Hash verification precedes stage continuation.
10. `SheetSage2Transcriber` orchestrates overlapping audio windows and score
    export. Its MERT2 encoder, adapter merge, BART decoder, and grammar live in
    the owned `MereRunSheetSageModel` target.

The native runtime does not invoke an external interpreter or execute model
repository code. It loads the pinned BF16 transformer and FP32 decoder directly
from safetensors. It rejects missing, extra, or incorrectly shaped consumed
tensors. The VAE encoder is intentionally excluded from generation.

The default configuration preserves upstream sampling, context boundaries,
guidance arithmetic, and 32 midpoint steps. Each stage resets its request-local
MLX random state. MLX and upstream PyTorch use different random generators; equal
seeds do not imply identical tokens or waveforms across runtimes.
Instrumental score planning uses the upstream helper's Intro, Verse, Chorus, and
Outro section tags. These planner-only tags do not become rendering lyrics.

## Validation

`Tests/MereRunCoreTests/Fixtures/YuE2` contains small random-weight outputs from
the pinned upstream CPU implementation. The exporter records its source
revision and library version. The original arithmetic reference remains
`0edaf2f4053ef4731334b8329834b107977f9637`; the upstream interface update
does not change these equations. These fixtures cover transformer caches, acoustic
velocity and integration, FP32/BF16 arithmetic, and full/tiled VAE output. They
contain no trained model weights.

The released checkpoints passed five bounded native generation runs on an
Apple M4 Max with 128 GB of memory: direct generation, a seeded repeat, full
score planning, and supplied-score generation. One vocal track ended naturally
at 36.08 seconds. The repeat was byte-identical. An optional installed-checkpoint
test also compares full and tiled decoding with the released VAE weights.

The 36.08-second sample also passed an owner listening review. These checks
establish bounded execution, audio integrity, and acceptance of one sample.
The November 7 M5 Max qualification adds a six-minute frame-limit run,
byte-identical saved-stage/direct and low-memory comparisons, real API
cancellation, released-weight CPU/GPU comparisons, and an owner-accepted
instrumental excerpt. Broader listening quality and device coverage remain
unqualified. See [the qualification report](../../../docs/runtime/yue2-qualification.md)
for the measured scope, memory observations, and instrumental planner prompt fix.

## Licenses

YuE2 source uses Apache 2.0; Oobleck and SnakeBeta retain their MIT notices in
`THIRD_PARTY_NOTICES.md`. The separately downloaded YuE2-3B and VAE model weights
use CC BY-NC 4.0. Managed installation requires explicit license acceptance.

## Interface and execution scope

`music yue2` exposes plan, semantic, synthesize, decode, inspect, validate,
instrumentalize, transcribe, cover, export, batch, and serve commands. Generation
can retain the same stages with `--stages-output`. Stage manifests store the
native RNG identity; same seeds do not reproduce PyTorch outputs.

The staged mode releases the song model before decoding. Resident mode retains
weights on one operation. Low-memory mode prefills a chunk with AR branches,
releases those branches and the output head, then loads NAR branches; it repeats
that exchange for later chunks. Prefix caches remain live. The VAE stays FP32.
Acoustic velocity evaluations reuse their key/value workspace. Performance and
full-song peak memory are recorded in the qualification report. The sampler
reclaims unused MLX buffers after prefill chunks and decoding steps when the
observed cache exceeds 512 MiB; active weights and KV state remain live.

SheetSage2 uses its specified parent checkpoint hash and FP32 adapter merge.
It preserves raw events when notation cannot be exported. The deterministic
score exporter supports the native 1/32 grid and bar-aligned meter changes;
this does not claim byte-identical serialization with upstream SheetSage2.
Independent upstream CPU fixtures cover the frontend, encoder, weighted layer
mix, adapter merge, BART prefill/cache, and grammar. Released-weight comparisons
and bounded production transcription now have recorded results. General
recording accuracy and musical quality remain unqualified.

PyTorch compilation, CUDA/XPU graph capture, vLLM request batching, and FP8 AR
are upstream backend mechanisms. Native MLX execution does not expose those
settings or claim their throughput. The local native API queues requests and
executes them serially. Completed stages remain on disk after cancellation.

`scripts/validation/qualify-yue2.py` measures separately supplied local
checkpoints with a fixed JSON request. It records stage timings, process memory,
decoder replay hashes, phase-loading latent differences, and finite-audio
checks. Optional inputs exercise the legacy VAE, transcription, and a long
generation request. It downloads no assets and makes no listening-quality claim.
