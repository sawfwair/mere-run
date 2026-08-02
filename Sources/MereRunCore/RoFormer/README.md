# RoFormer source separation

This module is the native Swift/MLX inference path for the pinned ViperX
BS-RoFormer checkpoint. It performs the STFT, axial time/frequency transformer,
complex masking, ISTFT, overlapped chunk aggregation, and vocal/instrumental
stem construction without launching Python.

The accepted checkpoint is the MIT-licensed
`AEmotionStudio/roformer-models` mirror at revision
`d323194290f8488ea51814143806609bfbd7a1e5`. `RoFormerResources` pins the exact
weights, source YAML, model-card README, and license. Do not accept another
checkpoint under this managed model ID without a fresh architecture, tensor,
provenance, and license audit.

Runtime configuration is typed and bundled at
`Resources/RoFormer/viperx-1297.json`. Keep it synchronized with the pinned
source YAML and cover geometry changes with tests.
