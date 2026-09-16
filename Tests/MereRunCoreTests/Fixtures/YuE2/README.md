# YuE2 numerical fixtures

These fixtures contain random initialized weights and CPU outputs from YuE2
source `0edaf2f4053ef4731334b8329834b107977f9637`, with PyTorch 2.10.0 and
Transformers 4.57.6. They contain no trained weights.

Run `scripts/fixtures/export-yue2-reference.py` with `--upstream` pointing to
that checkout and `--output` pointing to this directory. The script verifies
the source commit, uses CPU execution and seed 831001, and writes provenance.

The transformer has two 16-channel layers, two query heads, one KV head, and
the original 64-channel acoustic latent interface. Tests compare full and
cached autoregressive logits, cached acoustic velocity, and three midpoint
steps in FP32 and BF16.

The decoder uses two base channels and retains the original six strides.
Its 35-frame input exercises full and tiled decoding with eight-frame cores,
16-frame halos, and the natural 64-sample final crop.

`tokenizer-reference.json` contains IDs for the original string used in
`testCheckpointTokenizerWhenProvided`. It was generated with tiktoken 0.12.0
and `qwen.tiktoken` from pinned YuE2-3B revision
`29b3558dd46954a0cd9021dc76d5c91864a0f1c7`. The tokenizer artifact is not bundled.

Upstream source and component attribution appear in `THIRD_PARTY_NOTICES.md`.
