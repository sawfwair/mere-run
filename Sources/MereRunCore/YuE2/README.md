# YuE2

This directory owns native Swift/MLX inference for `music-yue2`. The implementation
follows YuE2 source revision `0edaf2f4053ef4731334b8329834b107977f9637`:

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
7. `YuE2GenerationOperation` owns stream lifetime, cancellation, staged loading,
   finite-output checks, and conversion to `AudioWaveform`.

The native runtime does not invoke an external interpreter or execute model
repository code. It loads the pinned BF16 transformer and FP32 decoder directly
from safetensors. It rejects missing, extra, or incorrectly shaped consumed
tensors. The VAE encoder is intentionally excluded from generation.

The default configuration preserves upstream sampling, context boundaries,
guidance arithmetic, and 32 midpoint steps. Each stage resets its request-local
MLX random state. MLX and upstream PyTorch use different random generators; equal
seeds do not imply identical tokens or waveforms across runtimes.

## Validation

`Tests/MereRunCoreTests/Fixtures/YuE2` contains small random-weight outputs from
the pinned upstream CPU implementation. The exporter records its source
revision and library version. These fixtures cover transformer caches, acoustic
velocity and integration, FP32/BF16 arithmetic, and full/tiled VAE output. They
contain no trained model weights.

The released checkpoints passed five bounded native generation runs on an
Apple M4 Max with 128 GB of memory: direct generation, a seeded repeat, full
score planning, and supplied-score generation. One vocal track ended naturally
at 36.08 seconds. The repeat was byte-identical. An optional installed-checkpoint
test also compares full and tiled decoding with the released VAE weights.

The 36.08-second sample also passed an owner listening review. These checks
establish bounded execution, audio integrity, and acceptance of one sample.
Broader listening quality, long-song memory and performance, and cancellation
during generation remain unqualified. See the model handbook for the measured
scope and limits.

## Licenses

YuE2 source uses Apache 2.0; Oobleck and SnakeBeta retain their MIT notices in
`THIRD_PARTY_NOTICES.md`. The separately downloaded YuE2-3B and VAE model weights
use CC BY-NC 4.0. Managed installation requires explicit license acceptance.
