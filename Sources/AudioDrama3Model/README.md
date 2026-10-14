# Native DRAMA-3 audio runtime

Owns typed checkpoint decoding, prompt rows, the persistent ten-codebook MTP tower,
and causal DAC encoder/decoder. `MereRunQwenModel/Drama3Backbone.swift` reuses the
Qwen3.5 attention, gated delta recurrence, and packed routed experts. AudioTTS
owns resource resolution, reference WAV preparation, and operation integration.

Original FP8 expert tensors are decoded with their 128x128 block scales and packed
into affine Q4/group-64 one expert at a time. All other model tensors retain BF16.
The codec executes in FP32 and is loaded with Core's non-executing restricted PyTorch state-dict reader.
The inference path does not launch Python or a remote server.

Architecture reference: SGLang-Omni at
`2cfb3b67becb437b70751defe37e63d3af4eedb0` (Apache-2.0).
Checkpoint: `fishaudio/drama-3@dd64a85cd82e9521f7202610ede97b367fc83946`.
Built with Fish Audio. Weights remain subject to the Fish Audio Community License.
