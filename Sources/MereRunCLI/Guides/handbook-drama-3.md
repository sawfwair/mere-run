# DRAMA-3 (Fish Audio)

`speech-tts-drama-3` runs the pinned Fish Audio DRAMA-3 checkpoint through
Swift/MLX. Built with Fish Audio. Review and accept the
[Community License](https://huggingface.co/fishaudio/drama-3/blob/dd64a85cd82e9521f7202610ede97b367fc83946/LICENSE)
before pulling the model.

## Example to adapt

Install the source, export Q4 experts, and synthesize a short scene:

```bash
mere.run model pull speech-tts-drama-3 --accept-model-license
mere.run model optimize speech-tts-drama-3 --output ./drama-3-q4
mere.run speech synthesize '<|speaker:0|> [sighs] Fine. [laughs] That seems fair.' \
  --model ./drama-3-q4 --seed 42 --output ./scene.wav
```

This example hasn't been validated with the released checkpoint. The source
weights and codec require approximately 41.5 GB. The standalone Q4 files require
approximately 27.4 GB, with additional space needed for runtime allocations.
The exporter preserves the source and requires a new output directory.

## Controls and variants

- The native loader packs routed experts as affine Q4/group-64. Other model
  weights stay BF16; the preserved DAC codec executes in FP32.
- Use `<|speaker:0|>` through `<|speaker:15|>` and inline event tags in the text.
- Use `--mode clone --ref-audio ./reference.wav --ref-text 'Exact transcript'`
  for one reference clip. References must be nonempty and at most 30 seconds.
- Use `--temperature` and `--seed` to control sampling.
- Use `--stream` and `--stream-chunk-tokens` for causal waveform prefixes.
  The codec recomputes the accumulated prefix for each chunk.
- Language is inferred from text. Use inline tags instead of `--voice`, named
  `--speaker` values, or classifier-free guidance.
- Generation has a 750-frame cap, approximately 35 seconds. Reaching the cap
  fails instead of publishing a truncated final WAV. Split longer text into
  shorter requests. Continuation editing isn't exposed through the CLI.
- The model is available through the local CLI. iOS retains metadata and rejects
  execution; the API server excludes this model, including local paths.

## Sources and validation

The checkpoint is pinned to `dd64a85cd82e9521f7202610ede97b367fc83946`.
The architecture reference is SGLang-Omni at
`2cfb3b67becb437b70751defe37e63d3af4eedb0`.

Component tests cover published tensor shapes, prompt boundaries, an independent
NumPy audio-tower reference, temporal caches, causal convolution, FP8 scales, and
Q4 export/reload with source preservation and hash receipts. The 32 GB minimum
and 36 GB recommendation are estimates. Released-checkpoint speech quality, Q4
quality loss, peak unified memory, and throughput remain unqualified.

- [Original checkpoint](https://huggingface.co/fishaudio/drama-3)
- [Native speech runtime](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/speech.md)
