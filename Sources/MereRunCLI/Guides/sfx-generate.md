# SFX Generate

## Purpose

Generate a WAV sound effect from a text prompt with a local Woosh text-to-audio
checkpoint stack.

## Models

- `sfx-woosh-dflow`: Sony Research Woosh distilled FlowMap model for fast
  text-to-SFX generation. The managed install pulls the T2A checkpoint, Woosh
  audio autoencoder, TextConditionerA weights, and RoBERTa tokenizer files.
- `sfx-woosh-flow`: Sony Research Woosh original Flow model for text-to-SFX
  generation. It uses the same Woosh-AE and TextConditionerA components but
  generally needs more denoise steps for quality.
- `sfx-woosh-clap`: Woosh-CLAP retrieval model for text/audio scoring.
- `sfx-woosh-dvflow-8s`: distilled video-to-audio model conditioned on
  raw video or Synchformer `synch_out` features.
- `sfx-woosh-vflow-8s`: original video-to-audio Flow model conditioned on
  raw video or Synchformer `synch_out` features.
- `sfx-woosh-synchformer`: companion visual extractor used by raw-video V2A
  generation.

## Commands

```bash
mere.run model pull sfx-woosh-dflow --accept-model-license
mere.run model pull sfx-woosh-flow --accept-model-license
mere.run model pull sfx-woosh-synchformer --accept-model-license
mere.run sfx generate --help
mere.run sfx clap score --help
mere.run sfx video generate --help
mere.run guide sfx generate --model sfx-woosh-dflow
```

## Key Options

- positional prompt: target sound-effect description.
- `--model`, `-m`: managed model id or local Woosh checkpoints root.
- `--duration`: output duration in seconds. The native path maps this to Woosh
  latent frames at 100 frames per second plus the terminal frame.
- `--steps`, `-s`: denoise steps. `sfx generate` runs 4 on both Woosh models
  unless told otherwise; the original Flow model generally needs more, so pass
  `--steps` for it. `sfx video generate` runs 4 on DVFlow and 32 on VFlow.
- `--cfg`: Woosh guidance scale. The upstream DFlow example uses 4.5.
- `--renoise`: one value repeated for all steps, or a comma-separated schedule.
  The default 4-step DFlow schedule is `0,0.5,0.5,0.3`. Only the distilled
  DFlow and DVFlow models renoise: Flow and VFlow check the schedule and print a
  warning that it has no effect, and MMAudio refuses it.
- `--negative-prompt`: MMAudio only; the CLI refuses it for Woosh.
- `--clip-batch-size` applies to MMAudio and `--synchformer-model` to Woosh; the
  other runtime prints a warning that it has no effect.
- `--seed`: deterministic MLX random seed.
- `--output`, `-o`: output WAV path.
- `--preflight`: for `sfx video generate`, inspect inputs, model requirements,
  output path, and denoise settings without loading MLX or generating audio.
- `--json`: with `--preflight`, emit a structured report with diagnostics and
  declarative follow-up actions.
- `sfx video generate` takes a raw video file by default. It can also take a
  feature `.npy` with shape `[frames, 768]` or `[1, frames, 768]`.
  Raw video inputs require `sfx-woosh-synchformer`; `.npy` feature inputs do not.

## Examples

```bash
mere.run sfx generate \
  "metal wrench dropping onto concrete, bright clang and brief ring" \
  --model sfx-woosh-dflow \
  --duration 5 \
  --steps 4 \
  --cfg 4.5 \
  -o wrench-clang.wav
```

```bash
mere.run sfx generate \
  "ceramic mug shattering on a tile floor, sharp cracks and scattered debris" \
  --seed 1234 \
  --renoise 0,0.5,0.5,0.3 \
  -o ceramic-shatter.wav
```

```bash
mere.run sfx generate \
  "dry branch snapping under a boot" \
  --model sfx-woosh-flow \
  --duration 1 \
  --steps 2 \
  -o branch-snap.wav
```

```bash
mere.run sfx ae encode branch-snap.wav -o branch-snap-latents.npy
mere.run sfx ae decode branch-snap-latents.npy -o branch-snap-roundtrip.wav
```

```bash
mere.run sfx condition text \
  "glass breaking" \
  -o glass-condition.safetensors
```

```bash
mere.run sfx clap score \
  "glass breaking" \
  glass.wav
```

```bash
mere.run sfx video generate \
  "footsteps echoing in a hallway" \
  silent-hallway.mp4 \
  --model sfx-woosh-dvflow-8s \
  -o hallway-footsteps.wav
```

```bash
mere.run sfx video generate \
  "footsteps echoing in a hallway" \
  silent-hallway.mp4 \
  --model sfx-woosh-dvflow-8s \
  -o hallway-footsteps.wav \
  --preflight \
  --json
```

## Notes

Each Woosh model runs one command. `sfx generate` takes DFlow, Flow, or MMAudio;
`sfx video generate` takes DVFlow, VFlow, or MMAudio; `sfx clap score` takes
Woosh CLAP. Naming another model stops at once and names the command it runs.
The CLI refuses an option the selected model doesn't support before it loads
anything.

Woosh is a sound-effect and Foley model, not a song-generation model, so it
lives under `sfx generate` instead of `music generate`. The managed weights are
the CC-BY-NC 4.0 open weights mirrored from Sony Research Woosh v1.0.0; respect
the non-commercial license when using generated outputs.

## Sources

- https://github.com/SonyResearch/Woosh
- https://huggingface.co/AEmotionStudio/woosh-models
- `Sources/MereRunCLI/Commands/SFXGenerateCommand.swift`
- `Sources/MereRunCore/Woosh/WooshGenerator.swift`
