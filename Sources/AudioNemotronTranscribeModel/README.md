# Nemotron 3.5 Transcribe native model

This target owns native Swift/MLX arithmetic for NVIDIA's released
`NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` BF16 checkpoint at revision
`ef905575a106fdad904f3b0ce661dccad654607d`. AudioSTT owns installed model
resolution, local audio decoding, tokenization, and transcription results.

The graph includes the 32-layer, 1280-channel ASR Transformer, half-head RoPE,
per-head Q/K normalization, eight-speaker Nemotron Sortformer branch, learned
speaker fusion, projection, and 52-block Nemotron-H language decoder. Mamba-2
retains FP32 recurrent state; attention has no positional rotation. Routed
experts use the original stacked `[expert, input, output]` matrices, sigmoid
scores, unbiased normalized mixing weights, and squared ReLU activations.
Prefill evaluates small expert gathers and each layer to bound Metal graphs.

`Resources/config.json` records inference fields from the pinned configuration;
`tensor-inventory.json` records original tensor shapes and dtypes. Loading
rejects architecture drift and incompatible parameters. Embedded CTC and extra
training state are omitted from the inference graph. No upstream Python,
conversion, vLLM service, or remote execution runs during inference.

The audio frontend uses the checkpoint's Hann window and mel bank, constant
STFT centering, preemphasis, and unbiased per-feature normalization. Audio is
split into independent 30-second waveform rows before feature extraction,
matching NeMo's `encode_audio_with_optional_chunking`; tails below 20 ms join
the previous row. Encoded rows share one language-model context and decode.
The speaker IDs are local to encoder rows, as in that reference recipe.

References: NVIDIA-NeMo/Speech revision
`1c1b25339ca2b31794ba4b299a5ea6a2c83c64b5`, `transformer_encoder.py`,
`parallel_expert_encoder.py`, `encoder_chunking.py`, and `nemotron3p5.py`.
Model configuration and inventory are NVIDIA model materials under OpenMDW-1.1.
See THIRD_PARTY_NOTICES.md. MTP and CTC timestamp heads require separate
qualification; this implementation exposes offline greedy text decoding.

## Installed-checkpoint regression

`NemotronTranscribeE2ETests` exercises the shared transcription operation over
the original BF16 weights and requires the exact known sentence. Set
`MERERUN_TEST_NEMOTRON_TRANSCRIBE_PATH` to the checkpoint directory and
`MERERUN_TEST_MLX_DEVICE=gpu`, then run `swift test --filter NemotronTranscribe`.
Use `MLX_ENABLE_TF32=0` for numerical reference comparisons. The recorded
2026-10-28 CLI qualification also covers the 46.57-second returning-speaker
fixture across the encoder chunk boundary; see
`docs/qualification/nemotron-transcribe-native-20261028/README.md`.
