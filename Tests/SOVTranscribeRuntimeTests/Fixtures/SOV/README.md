# SOV numerical references

`reference.json` contains deterministic FP32 tensors generated independently
with NumPy by `scripts/generate-sov-fixtures.py`. It covers Whisper convolution,
layer normalization, attention, exact GELU, the stacked-frame projector, and a
DFlash model with causal sliding attention followed by bidirectional attention.
It includes five target-context positions and four draft positions. Context
append tests cross a three-position sliding window.

Regenerate from the repository root with Python and NumPy installed:

```bash
python3 scripts/generate-sov-fixtures.py
```

`released-config.json` and `released-draft-config.json` are the public Salesforce
configuration files at revisions `505d8fae243924c0d8acdc622f7f0abab344eba3`
and `8c0b3f3a98507d2e3dd35b1bee0f147fa3bded05`. Their sources are linked in
`docs/runtime/sov-transcribe.md`. No checkpoint weights or audio recordings are
included. These fixtures verify arithmetic and contracts, not released-model
transcription quality or hardware performance.
