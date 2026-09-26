# Breeze TTS 2 native runtime

This target owns the Swift/MLX Breeze TTS 2 text encoder, acoustic backbone,
depth decoder, and synthesis loop. It reuses `AudioQwen3TTSModel` for the
checkpoint's separate `audio_tokenizer/` encoder and decoder. `AudioTTS` owns
model selection, reference audio preparation, and WAV publication.

The model source and checkpoint have separate licenses. The upstream inference
source is Apache-2.0. Portions of this Swift graph were adapted from
`Blaizzy/mlx-audio-swift` under MIT. BreezeBlue checkpoint weights and
self-hosted outputs are limited to research and non-commercial use. No weights
are stored in this repository.
