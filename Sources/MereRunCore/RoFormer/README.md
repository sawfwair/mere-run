# RoFormer source separation

This module is the native Swift/MLX inference path for pinned ViperX and
four-stem BS-RoFormer checkpoints. It performs the STFT, axial time/frequency
transformer, complex masking, ISTFT, and overlapped chunk aggregation without
launching Python.

The accepted checkpoint is the MIT-licensed
`AEmotionStudio/roformer-models` mirror at revision
`d323194290f8488ea51814143806609bfbd7a1e5`. `RoFormerResources` pins the exact
weights, source YAML, model-card README, and license for both admitted model
profiles. Do not accept another checkpoint under either managed model ID
without a fresh architecture, tensor, provenance, and license audit.

Runtime configuration is typed and bundled at
`Resources/RoFormer/viperx-1297.json`. Keep it synchronized with the pinned
source YAML and cover geometry changes with tests.
