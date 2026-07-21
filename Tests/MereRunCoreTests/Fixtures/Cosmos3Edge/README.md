# Cosmos3-Edge parity fixtures

These fixtures were exported from NVIDIA's `cosmos-framework` at commit
`ed8287fd7477113f8ac4f6b84290514d55cf0cdc` against `nvidia/Cosmos3-Edge`
revision `6f58f6b4c91288838e60b6bcb2cc45d997e961de`.

- `tiny-mot-layer.safetensors` contains deterministic inputs, weights, mRoPE
  values, and outputs for a one-layer Nemotron MoT block. It specifically
  covers raw understanding Q/K, normalized generation Q/K, the separately
  normalized understanding K used by generator cross-attention, and ReLU².
- `tiny-reasoner-vision.safetensors` pins the packed SigLIP2 attention tower,
  interpolated 2D positions, NVIDIA's raster patch convention, 2x2 spatial
  merge order, and multimodal projector.
- `checkpoint-inventory.json` records all published transformer and VAE tensor
  names, shapes, and dtypes without embedding model weights.
- `RECEIPT.json` records the pinned revisions and fixture digests.

Regenerate with `scripts/export-cosmos3-edge-parity.py` from the pinned NVIDIA
checkout. The upstream code and published model are licensed under NVIDIA's
Open Model Development and Open Model licenses respectively; the fixtures are
small numeric interoperability/conformance data, not distributed model weights.
