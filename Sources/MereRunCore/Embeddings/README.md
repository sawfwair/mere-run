# Text embeddings

`EmbeddingGemma2Resources.swift` owns the pinned managed checkpoint and task
prefixes. `EmbeddingGemma2Model.swift` owns tokenizer preparation, typed config
validation, text-only checkpoint loading, and bounded batching. Native encoder
math lives in `MereRunGemmaModel/EmbeddingGemma2TextModel.swift`,
`EmbeddingGemma2VisionModel.swift`, and `EmbeddingGemma2AudioModel.swift`.

`Qwen3EmbeddingResources.swift` and `Qwen3EmbeddingModel.swift` retain the Qwen3
default. The shared length-based batch packer restores caller order and bounds
padded tokens. EmbeddingGemma 2 uses mean pooling including task/BOS/EOS tokens;
Qwen3 pools the final valid token. Keep those semantics separate.

`EmbeddingGemma2Model+Media.swift` expands ordered image, audio, and video
blocks and replaces placeholder tokens with projected native tower outputs.
`EmbeddingGemma2MediaProcessor.swift` prepares bicubic RGB patches and semicausal
log-mel features through `MediaIO`. Video uses 1-fps sampling capped uniformly
at 32 frames. Media records run sequentially and refuse partial-block truncation.
The vision/audio towers load lazily; text-only requests never instantiate them. Model installation downloads the original
complete safetensors artifact. No checkpoint conversion or external inference
process is required.
