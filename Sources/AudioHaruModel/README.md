# Native Haru audio runtime

This target implements Haru-4B's frame protocol, audio projections, MeanFlow
head, and stateful VibeVoice acoustic codec in Swift and MLX. The shared
Qwen3.5 backbone evaluates one position per 3,200 samples of mono 24 kHz PCM.
Model text and the two audio latents are summed before the backbone. Each
session owns its text history, random stream, attention caches, and causal
codec state. Voice reference encoding uses independent codec state.

The loader accepts the original PyTorch-layout safetensors. It checks active
parameter coverage and keeps the backbone's zero-centred RMS gains in FP32.
Architecture reference is higurashi-ai/Haru at
`7fe6b7665f8bf035998f401f8cb779fbe53f2751` (Apache-2.0).
