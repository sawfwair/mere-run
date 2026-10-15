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

The Q4 export has been validated with the released checkpoint: 27.41 GB across
99 files, with a 2.62 GB peak physical footprint during conversion. The scene
generation example still requires speech qualification. The source weights and
codec require approximately 41.5 GB. Inference needs additional memory.
The exporter preserves the source and requires a new output directory. Use
`model location bind speech-tts-drama-3 /path/to/drama-3-q4 --accept-model-license`
to register a read-only checkpoint on an external volume.

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
Q4 export/reload with source preservation and hash receipts. The released DAC
matches all 220 encoded positions in a synthetic reference probe; deterministic
waveform decoding has RMSE 0.0001933 against the pinned upstream graph.
The 32 GB minimum
and 36 GB recommendation are estimates. Released-checkpoint speech quality, Q4
quality loss, peak unified memory, and throughput remain unqualified.

- [Original checkpoint](https://huggingface.co/fishaudio/drama-3)
- [Native speech runtime](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/speech.md)
