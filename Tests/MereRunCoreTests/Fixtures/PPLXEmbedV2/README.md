# PPLX Embed v2 synthetic references

`export-pplx-embed-v2-reference.py` exports a tiny, seeded FP32 Qwen3.5 model
using Transformers 5.4.0 and Torch. It contains one causal gated-delta layer
and one bidirectional full-attention layer, plus a small vision tower. Offset
RMS weights are deliberately nonzero. `reference.json` contains original tensor
layouts and independently computed text, image-tower, and multimodal outputs.

The byte-level BPE tokenizer is trained on a tiny fixed multilingual corpus.
`tokenizer-cases.json` records independent Rust tokenizer results for prefixes,
punctuation, UTF-8, literal special tokens, and empty/chunk-boundary spans.
It is used only for tests; production always loads the checkpoint tokenizer.

Regenerate from the repo root with Torch, tokenizers 0.22.2, and
Transformers 5.4.0. No weights or tokenizer assets are downloaded.

These fixtures establish component parity, not full-checkpoint retrieval
quality, cross-model embedding alignment, or 262144-token memory qualification.

The separate `real-tokenizer-nfc-cases.json` and
`real-context-tokenizer-nfc-cases.json` record Transformers 5.4.0 token IDs and
original-text offsets from the pinned late/context 9B tokenizers. Both apply
NFC. The real context separator is a non-special added token and survives
`split_special_tokens=True`; the tiny synthetic tokenizer marks it special.
The native regression covers both distinctions. No real vocabulary or weights
are committed here.

Regenerate these two files with
`scripts/fixtures/export-pplx-embed-v2-tokenizer.py --kind late|context --source
/path/to/pinned-checkpoint --output /path/to/fixture.json`. Run the opt-in test
with `MERERUN_PPLX_TOKENIZER_ROOT=/path/to/pinned-checkpoint swift test --filter
PPLXEmbedV2Tests.testReal9BTokenizerNFCOffsetsWhenCheckpointIsAvailable`.
