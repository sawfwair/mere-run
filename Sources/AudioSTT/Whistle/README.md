# Whistle transcription

This directory resolves the pinned original checkpoint, reads audio through
AudioCodecs, extracts Whistle features, and coordinates native beam search and word alignment.
Model arithmetic and binary vocabulary decoding live in `AudioWhistleModel`.

The frontend uses the published 257-by-80 filterbank, a symmetric 400-sample Hann
window, a 512-point FFT, a 160-sample hop, centered zero padding, natural log,
and channel-wise sample variance. It is not interchangeable with Whisper's
frontend. Files longer than 30 seconds use one-second overlapping windows and
word overlap reconciliation. Each window has its own decoder state.

The first step selects one of seven language tokens; subsequent steps decode text
until EOS or the token budget. Typed controls select 1...8 beams, 2...8 decoder
layers, keyword biasing, and cross-attention DTW word alignment. Packed CQ2/CQ4
weights are the default; original FP32 safetensors are an explicit local baseline.

`WhistleGenerator` keeps one model resident for `ASRUtteranceLiveSession`. Loading
and prepared decoding use the shared `MLXRequestStreams` pool, with a genuine
async yield before constructing/evaluating graphs. Materialize weight tensors
before returning the loading lease. No lazy inference output may escape its
stream lease. Cancellation is checked at load, window, encoder-layer, and decoder
step boundaries; submitted GPU work finishes before a lease is reused.
Routing rejects unsupported tasks, languages, and execution providers before load.
