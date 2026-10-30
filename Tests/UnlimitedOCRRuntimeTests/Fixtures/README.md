# Unlimited-OCR-2 fixtures

`config.json` is the published configuration from `baidu/Unlimited-OCR-2` at
`c14071b549f4fcab10909f3c4acc6a2edb366947` (MIT).

`attention.json` contains deterministic FP32 inputs and outputs from an
independent scalar implementation of the upstream DASH inference equations.
It uses 41 tokens, two prompt pages, two response pages, two-token reference
blocks, a four-token local window, a top-three entmax window, and two selected
remote blocks. These small dimensions exercise transitions without model weights.
The fixture generator is `generate_reference.py` and requires only Python's
standard library. These fixtures don't establish checkpoint OCR quality.

`decoder.json` uses two decoder layers, eight hidden channels, four routed
experts, and two active experts per token. The scalar reference computes causal
attention, page RoPE, dense SwiGLU, unnormalized top-k softmax MoE plus shared
experts, RMSNorm, and final logits. Native tests compare full prefill and
incremental cache steps with these logits.
