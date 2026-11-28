# RoFormer source separation

This module is the native Swift/MLX inference path for pinned ViperX and
four-stem BS-RoFormer checkpoints plus MelBand RoFormer dereverb and denoise.
It performs the STFT, axial time/frequency transformer, complex masking, ISTFT,
and overlapped chunk aggregation without launching Python. MelBand profiles
reproduce librosa's Slaney mel supports and average masks at overlapping bins.

The accepted checkpoint is the MIT-licensed
`AEmotionStudio/roformer-models` mirror at revision
`d323194290f8488ea51814143806609bfbd7a1e5`. `RoFormerResources` pins the exact
weights, source YAML, model-card README, and license for every admitted model
profile. Do not accept another checkpoint under an existing managed model ID
without a fresh architecture, tensor, provenance, and license audit.

Runtime configurations are typed and bundled under `Resources/RoFormer`.
Keep them synchronized with the pinned source YAML files and cover geometry
changes with exact parameter-inventory and frequency-layout tests.
