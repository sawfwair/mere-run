# Phonon 31m (Gradium)

Use Phonon 31m to generate English speech locally with a small, Apache-2.0
checkpoint and its bundled voice. The managed model is `speech-tts-phonon-31m`.

## Example to adapt

```bash
mere.run model pull speech-tts-phonon-31m
mere.run speech synthesize "Hello from mere run." \
  --model speech-tts-phonon-31m --seed 42 -o phonon.wav
mere.run speech synthesize "Speech can begin before generation finishes." \
  --model speech-tts-phonon-31m --stream -o phonon-stream.wav
```

The native Swift/MLX runtime loads the original safetensors directly. It emits
mono audio at 24 kHz. Offline WAV files use PCM16; streaming WAV files use
float32. Streaming emits one codec frame (80 ms) at a time; discrete token
counts do not apply. Omit `--stream-chunk-tokens`, which Phonon rejects.

Select a local checkpoint folder with `--model /path/to/phonon-31m-en`. The
folder must contain `config.json`, `model.safetensors`, `tokenizer.json`, and
`default-voice.safetensors`. Model loading checks tensor shapes and the voice's
recorded compatibility metadata. Use a separate local folder to substitute a
compatible precomputed default voice.

## Controls and variants

Use `--language en` or leave language detection at `auto`. `--temperature`
controls latent noise; `--seed` makes sampling repeatable within this runtime.
Seeds do not promise identical random samples across Gradium's Rust runtime
and MLX. The default temperature remains the speech command's 0.6; Gradium's
checkpoint recommends 0.8, which you can select explicitly.

Phonon 31m does not provide voice instructions, named speakers, raw-audio voice
cloning, classifier-free guidance, or speed changes through this adapter.
Omit `--voice`, `--speaker`, and `--cfg-scale`, and use style mode. The API
accepts `model: "speech-tts-phonon-31m"` with an omitted or `"default"` voice,
no instructions, and speed 1.

Spell out numbers, dates, email addresses, and other structured values before
synthesis. This implementation tokenizes and prepares English text but does
not reproduce Gradium's structured-text rewrite rules. Long text is fitted to
sentence-aligned chunks of at most 200 tokens using Gradium's token-based
planner. Each chunk starts from the voice prefix with fresh text, attention,
and codec state. A
frame-budget overrun is an error, so an incomplete utterance is not published
as a successful WAV.

## Sources and validation

- Checkpoint: [gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en/tree/92741c5b07d085ff840e26fea8ce43110b87852f).
- Source contract: [Gradium xn-ptts](https://github.com/gradium-ai/xn-ptts/tree/49b9033e44ab0fbb935973eca3213757644d1bbb).
- Checkpoint weights, tokenizer, and bundled voice: Apache 2.0.
- Adapted runtime algorithms: MIT; see `THIRD_PARTY_NOTICES.md`.

The managed download is about 103 MB. This runtime loads the original mixed BF16/F32 checkpoint and computes in
F32. The launch-page footprint is not the download size of this snapshot. It does not load the GGUF files.

Released-weight qualification compares a 24-case shared-noise corpus against
the pinned upstream F32 runtime, including frame counts, waveform similarity,
latent/EOS traces, local ASR intelligibility, and warmed release timings.
Native Metal is faster than real time on the measured M3 Max, while upstream's
optimized CPU paths remain faster in total generation time. The comparison
records voice-caching and pipeline differences; it does not establish the
publisher's performance claims or blind listener preference. See
`docs/architecture/phonon-native-qualification.json` and
`scripts/phonon-qualification/README.md` for evidence and reproduction steps.
The gated 40m checkpoints and multilingual models are not included.

CUDA checkpoint comparisons use full F32 math with `MLX_ENABLE_TF32=0` and
disable CUDA graphs with `MLX_USE_CUDA_GRAPHS=0`. Set these variables before
launching the CLI. The recorded upstream numerical tolerances do not cover
TF32 math. See `docs/qualification/phonon-cuda-2026-10-23.json` for the bounded
Linux CUDA follow-up.
