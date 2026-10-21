# Flint test evidence

`config.json` and `chat_template.jinja` come from `RecursiveSI/flint` at
`2a247be0c0aec895ccb0ac7ca92229ec6df37288`. `int4-config.json` comes from
`RecursiveSI/flint-int4` at `3b93f9053b530488405ae9fca51d4900d6ce21f0`.
Their Apache-2.0 license is retained in `LICENSE.Apache-2.0`.

`int4-layout.json` records only tensor names, dtypes, and shapes from the
safetensors headers of the five pinned INT4 shards. Header acquisition used
HTTP byte ranges for the eight-byte header length and the JSON header body.
It contains metadata for all 1,751 published tensors and no checkpoint values.
The runtime layout test checks every entry, including the U8-to-U32 storage
reinterpretation and the INT8 embedding policy.

`generate-reference.py` generates the tiny configuration, both synthetic
checkpoint layouts, source index, and reference logits using Python's standard
library. It does not import any native model code or MLX. Its three layers
include full and sliding attention, learned sinks, dense and routed feed-forward
blocks, shared experts, and selection-only probability bias. The reference
computes the complete causal sequence with scalar arithmetic. Native tests
compare full and chunked execution across more than two sliding windows.

These fixtures establish synthetic arithmetic and structural layout agreement.
They do not establish released-weight inference parity, model quality, memory,
throughput, or long-context accuracy.
