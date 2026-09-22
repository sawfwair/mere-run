# MereRunLayaModel

Native Swift/MLX ModernBERT and Laya decision computation. Core owns tokenizer
loading, model resolution, request validation, batching, and result formatting.

The encoder uses alternating global and bidirectional local attention, full-head
split-half RoPE, exact GELU gating, and the first-layer attention-norm omission.
The decision head uses pre-norm attention and ReLU feed-forward layers. The
option scorer and action head use exact GELU. All computation uses float32,
matching the upstream CPU/MPS evaluation path.

Original safetensors names and shapes are checked before inference. No Python,
ONNX, conversion, or remote model code runs in the native inference path.
