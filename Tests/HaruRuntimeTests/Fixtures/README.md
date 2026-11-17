# Haru numerical fixture

`model.safetensors` contains synthetic FP32 weights generated with NumPy's
default RNG and seed 731. Its tensor names and layouts follow the published
Haru-4B PyTorch checkpoint at `8d6e83bcd3701a76fc27a764965665da1df7da78`.
`config.json` reduces the model width and depth while retaining both Qwen
attention types and every codec stride.

`reference.safetensors` was exported on CPU using the publisher's MLX
implementation at `7fe6b7665f8bf035998f401f8cb779fbe53f2751`. It contains
four prefill positions followed by two single-position decodes, audio and
text projection inputs, and a fixed MeanFlow noise input. Codec references
use a partial final input frame and decode two consecutive latents.
