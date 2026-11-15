# SOV transcription orchestration

`SOVTranscribeGenerator` prepares SOV-Transcribe-1 and its optional DFlash
companion, then runs local transcription through MLX or a Core ML audio encoder.
`AudioSOVTranscribeModel` owns audio and draft computation;
`MereRunGemmaModel` owns the target decoder and expert kernels.

## Reading order

1. `SOVTranscribeGenerator.swift`: request validation, model preparation,
   stream ownership, generation, and unload.
2. `SOVTranscribeWeights.swift` and `SOVPackedWeights.swift`: checkpoint loading,
   streamed expert quantization, packed checkpoint validation, and direct reuse.
3. `SOVTranscribePreprocessing.swift`: audio features and target prompts.
4. `SOVCoreMLEncoder.swift`: optional compiled encoder/projector validation and
   execution. The target and DFlash decoder stay on MLX.
5. `SOVDecoderExperts.swift`, `SOVGenerationExperiment.swift`, and
   `SOVBenchmark.swift`: explicit research policies and measured inference.

## Runtime rules

Preparation replaces expert modules before decoding. Packed checkpoints contain
quantized target parameters; dispatch policy is independent of the saved format.
Eligible Q8 Metal projections select distributed reduction automatically on the
qualified M5 Max GPU. Other devices and unsupported shapes keep the existing
kernel. This selection does not change the requested expert precision.

Use typed decoding at checkpoint, manifest, and tokenizer boundaries. Keep
request execution inside `MLXRequestStreams`; benchmark policy switches preserve
resident parameters and record the selected kernel. Audio-token trimming,
alternate quantizers, and activation fusion retain their explicit controls.

## Validation

Run `swift test --filter SOV` and the focused Gemma expert tests, then
`./scripts/check.sh` before opening a PR. Released-checkpoint parity and timing
require real assets on an Apple GPU. See the
[runtime guide](../../../docs/runtime/sov-transcribe.md) and its qualification
reports for commands, hardware scope, frozen-output checks, and measured results.
