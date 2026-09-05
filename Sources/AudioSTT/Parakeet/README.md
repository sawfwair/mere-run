# Parakeet STT

Parakeet speech-to-text backend implementation.

- `ParakeetConfig.swift`: typed model configuration.
- `ParakeetTokenizer.swift`: tokenizer loading and text decoding.
- `ParakeetModel.swift`: native model layers.
- `ParakeetGenerator.swift`: transcription orchestration.
- `ParakeetExecutionProvider.swift`: explicit MLX or Core ML provider selection
  and standalone package discovery.
- `ParakeetCoreMLEncoder.swift`: verified Core ML/MLX artifact contract and
  encoder tensor bridge.
- `ParakeetCoreMLDecoder.swift`: batched TDT decoder, recurrent state, and
  embedding-table bridge.
- `ParakeetCoreMLDecoderOutput.swift`: returned Core ML tensors and stride-aware
  decoder state transfer, including exact base-128 FP16 token decisions.
- `ParakeetBenchmark.swift`: resident pipeline stage timings.
- `ParakeetCoreMLWindowing.swift`: quiet-boundary selection, bounded overlap,
  and global alignment offsets for the static-shape Core ML encoder.

Keep backend routing in `AudioCore`; this directory should only own Parakeet
loading, inference, and decoding behavior.

Schema-v4 artifacts use ANE-compatible encoder masks and decoder selection.
Use `scripts/model-conversion/inspect_parakeet_coreml.py --require-ane` and an
Instruments trace to qualify placement on the target hardware. Runtime support
for schemas 1 through 3 preserves existing artifact compatibility.
