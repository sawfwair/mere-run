# Unlimited-OCR-2

Native MLX computation for Baidu's Unlimited-OCR-2: SAM/CLIP vision,
DeepSeek sparse MoE decoding, page coordinates, and DASH reference attention.
Core owns checkpoint loading, image preparation, tokenization, and generation.

Reference: `baidu/Unlimited-OCR-2` at
`c14071b549f4fcab10909f3c4acc6a2edb366947` (MIT; inherited DeepSeek code is Apache-2.0).
The runtime reads original BF16 safetensors; it executes no upstream code.

Activation and scalar operations preserve PyTorch's BF16 rounding boundaries.
Prefill attention uses the query dtype for operands and bias; incremental
decode uses FP32 math before returning BF16. Package-scoped block functions and
optional observers support isolated upstream tensor diagnostics in the tests.
DASH combines inverse-square-root and per-layer scalars in double precision
before their FP32 tensor multiplication, matching Python scalar promotion.
Independent goldens cover the released 128-dimensional router.

Optional package-scoped activation substitutions support causal comparisons
against externally captured upstream states. They default to no substitution.
The layout diagnostic replays the common receipt prefix, probes blocks on
identical inputs and caches, and checks the first differing projection cells
against CPU FP64 accumulation. It doesn't change production model precision.
