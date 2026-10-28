# DRAMA-3 fixtures

`config.json` and `nonexpert-shapes.json` are metadata from
`fishaudio/drama-3@dd64a85cd82e9521f7202610ede97b367fc83946`.
Only safetensors headers and ZIP metadata were fetched; no full weights are included.

`tower-reference.json` is an independent NumPy Float32 evaluation of a three-row
causal GQA tower layer, with zero-centered norms, partial interleaved rotary,
output gates, semantic masks, code embeddings, and the final normalization.
Parameter values are `sin((index + sum(UTF8(name)) % 97) * 0.17) * 0.03`.
The reference uses NumPy einsum and explicit causal softmax. It does not import
Swift, MLX, or the upstream serving code. The fixture generator is retained in
`generate-tower-reference.py` for reproducibility.

These fixtures establish structural and component arithmetic coverage, not
released-checkpoint speech quality or laptop memory qualification.
