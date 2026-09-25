# AuK (Tencent Hunyuan)

## Purpose

Generate or edit speech with natural-language instructions and optional reference audio.

Use this guide to run the experimental native Swift/MLX AuK pipeline. The runtime
supports instruction-conditioned generation and reference-audio editing. Base
and Flash passed trained-weight parity and eight bounded audio checks each on
an M4 Max. See `docs/benchmarks/auk-native-qualification-2026-09-18.md` for results.

## Install the checkpoints

```bash
mere.run model pull audio-auk-base
```

`audio-auk-base` and `audio-auk-thinker` are both required for base inference.
The base pull also installs the separate Qwen2.5-Omni-3B encoder. Its download
includes the two original shards containing Thinker text and audio tensors,
omitting the unused third shard. The runtime loads only the tensors it needs.
The catalog specifies 24 GB minimum and 32 GB recommended unified memory.
Bounded qualification runs peaked at 18.4 GB, measured on a 128 GiB machine;
longer clips can require more memory.

To use the distilled model, pull `audio-auk-flash` and select it with `--model`.
Base and Flash have distinct fusion weights; do not rename or mix checkpoints.

## Example to adapt

```bash
mere.run audio edit "Say 'Welcome to the workshop' in a calm, clear voice." \
  --model audio-auk-base --duration 4 --output welcome.wav
```

Text-only requests require `--duration`. Output is mono float32 WAV at 24 kHz.
Duration rounds up to the model's 480-sample latent grid (20 ms).

## Controls and variants

```bash
mere.run audio edit "Remove background noise while preserving the speaker." \
  --model audio-auk-base --audio input.wav --output clean.wav

mere.run audio edit "Say 'Welcome back' with the same voice." \
  --model audio-auk-flash --audio reference.wav --duration 3 --output cloned.wav
```

Omit `--duration` to follow the encoded reference length. Audio is decoded to
mono at 24 kHz for the VAE and 16 kHz for the Qwen encoder. Reference clips must
be between 25 ms and 300 seconds. Base defaults to 32 Euler steps with guidance
2. Flash always uses its trained four-step grid and disables guidance, so a
`--steps` other than 4 or a `--guidance` other than 0 prints a warning that it
has no effect with Flash.

For existing downloads, pass `--model-path /path/to/AuK` and
`--thinker-path /path/to/Qwen2.5-Omni-3B`. The runtime reads original safetensors
in Swift; it does not invoke Python or require a conversion script.

## Sources and validation

The implementation uses sequential model loading and deterministic reference
posterior means. Same-seed output does not promise parity with PyTorch's random
number generator or its stochastic reference encoder. Qualification covers
English and Chinese speech content, reference-conditioned generation, phrase
replacement, pitch, speed, volume, and an enhancement proxy. Editing controls
are approximate: the report includes measured deviations. ASR content checks
do not measure voice identity or naturalness, and long clips and other editing
tasks remain outside the tested scope. The command reports experimental status
in JSON.

AuK code and weights use the MIT license. The Qwen encoder has its own license
in its upstream repository. See `THIRD_PARTY_NOTICES.md` for source attribution.
