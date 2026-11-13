# Native Inkling 2 computation

This target owns the Inkling 2 decoder, vision transformer, checkpoint byte
reader, and original NVFP4/MXFP8 expert decoding. Core owns tokenization,
media loading, generation, and model discovery.

The reference path reads selected experts from the original safetensors shards
and retains ordinary weights in memory. It expands one selected expert at a
time. Expert activations use BF16, with FP32 routing and accumulation; this is
a weight-dequantized reference path, not the upstream low-precision activation
kernel. Disk throughput limits performance. MTP is not used.

Architecture reference: Hugging Face Transformers `models/inkling2`, Apache-2.0.
Checkpoint: `thinkingmachines/Inkling-2-Small` at
`aac974122acd33ddb0d3e1f54433a9c5dd2249ce`.
