# Native Phonon

`AudioPhononModel` owns the Swift/MLX English Phonon flow language model,
Lagrangian self-distillation head, streaming Mimi decoder, and tokenizer.
`AudioTTS/Phonon` owns model resolution and the shared synthesis adapter.

The source contract is Gradium's `xn-ptts` at
`49b9033e44ab0fbb935973eca3213757644d1bbb`. The managed checkpoint is
`gradium/phonon-31m-en` at `92741c5b07d085ff840e26fea8ce43110b87852f`.
Original mixed BF16/F32 safetensors load directly and evaluate in F32. No
checkpoint conversion or external inference process is required.

Only English, a bundled or precomputed voice, and unguided generation are
supported. Raw-audio voice cloning, text-described voices, GGUF quantization,
40m checkpoints, and the upstream structured-text rewrite pipeline are outside
this implementation. Spell out structured values before synthesis.

The released checkpoint has been compared against pinned upstream F32 output
using shared noise on a 24-case corpus. See
[`phonon-native-qualification.json`](../../docs/architecture/phonon-native-qualification.json)
and [`scripts/phonon-qualification`](../../scripts/phonon-qualification) for
measurements, artifact hashes, reproduction steps, and scope. Numerical
agreement and ASR intelligibility do not establish listener preference.

On Linux, split attention projections are materialized in contiguous head-major
layout before CUDA attention consumes them. Full F32 comparisons use
`MLX_ENABLE_TF32=0` and `MLX_USE_CUDA_GRAPHS=0`; TF32 math does not meet the
recorded upstream tolerances. See the bounded
[`CUDA qualification`](../../docs/qualification/phonon-cuda-2026-10-23.json).
