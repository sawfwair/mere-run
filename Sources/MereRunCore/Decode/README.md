# Qwen prompt enhancement

`QwenGeneration.swift` adapts the shared Qwen text encoder and tokenizer to
prompt enhancement. Core owns the prompt template, tokenizer, encoder cache,
and generation configuration.

Shared sampling, token streaming, decode loops, and logprob types live in
`MereRunDecode`. Attention caches, including the optional affine quantized
cache, live in `MereRunKVCache`. Core re-exports both libraries.

Read each library's README before changing scheduling or cache behavior.
