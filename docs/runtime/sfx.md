# SFX runtime

Generate Foley from a sentence or silent video. Describe the sound you want to
create, or provide a clip so the runtime can align the generated sound with the
on-screen action.

## Commands

| Command | What it does |
| --- | --- |
| `mere.run sfx generate` | Generate a sound effect from a text prompt. |
| `mere.run sfx video generate` | Generate an 8-second sound effect from a video or Synchformer features. |
| `mere.run sfx clap score` | Score a text prompt against an audio file, to rank takes. |
| `mere.run sfx condition text` | Encode a prompt with the Woosh text conditioner. |
| `mere.run sfx ae encode` | Encode audio into normalized Woosh-AE latents. |
| `mere.run sfx ae decode` | Decode normalized Woosh-AE latents into audio. |

## Model families

- `sfx-woosh-dflow`
- `sfx-woosh-flow`
- `sfx-woosh-clap`
- `sfx-woosh-synchformer`
- `sfx-woosh-dvflow-8s`
- `sfx-woosh-vflow-8s`
- `sfx-mmaudio-large-44k-v2`

## macOS SFX Lab

The Sound FX workspace opens **SFX Lab**, a dedicated surface for all six
command families. It provides text generation and video-synchronized Foley,
renoise and negative-conditioning controls, text-conditioning export, Woosh-AE
encode/decode, and CLAP prompt/audio scoring. Generated audio and source video
are reviewable in place. Encoded NPY results show their data type, shape, NumPy
version, layout, and size. Progress and every output remain durable in Library.

## Writing prompts

Woosh is trained on the AudioCaps, WavCaps, and Freesound caption datasets, where
every label is a natural-language clause describing an event, and Sony's own
examples follow that form ("A crowd applauds", "An engine humming and brakes
squealing"). Keyword lists usually work too, but caption phrasing is more
reliable, and for some sounds it is the difference between audio and silence.

Measured on `sfx-woosh-dflow`, seed 7, peak amplitude of the generated WAV:

| Prompt | Peak |
| --- | --- |
| `grand piano lid closing` | 0.015 — inaudible |
| `grand piano lid closing, soft thud, long room tail` | 0.003 — inaudible |
| `A piano lid closing` | 1.000 |
| `A grand piano lid closes with a soft thud` | 0.573 |
| `ceramic mug shattering on a tile floor` | 0.082 — very quiet |
| `A ceramic mug shatters on a tile floor` | 0.893 |

The subject does not cause the collapse. `piano` and `lid closing` each generate
audible output independently, and `piano lid closing, loud` recovers. Sampling
also does not cause the collapse. The seed, `--cfg`, and `--steps` do not change
the result, and the `dflow` and `flow` checkpoints fail identically on the same
prompts.

When a generation comes out inaudible, `sfx generate` warns with the measured
peak. Rephrase as a caption before reaching for other parameters:

```bash
# Collapses to silence
mere.run sfx generate "grand piano lid closing, soft thud, long room tail"

# Generates cleanly
mere.run sfx generate "A grand piano lid closes with a soft thud"
```

Naming a louder variant of the same event ("slamming shut" rather than
"closing") also recovers a quiet prompt.

## Guides

```bash
mere.run guide sfx generate --model sfx-woosh-dflow
```

## Typical workflow

```bash
swift run mere.run model pull sfx-woosh-dflow --accept-model-license
swift run mere.run sfx generate \
  "metal wrench dropping onto concrete, bright clang and brief ring" \
  --model sfx-woosh-dflow \
  --duration 5 \
  --steps 4 \
  --cfg 4.5 \
  --output ./wrench-clang.wav
```

```bash
swift run mere.run model pull sfx-woosh-flow --accept-model-license
swift run mere.run sfx generate \
  "dry branch snapping under a boot" \
  --model sfx-woosh-flow \
  --duration 1 \
  --steps 2 \
  --output ./branch-snap.wav
```

```bash
swift run mere.run sfx generate \
  "ceramic mug shattering on a tile floor, sharp cracks and scattered debris" \
  --seed 1234 \
  --renoise 0,0.5,0.5,0.3 \
  --output ./ceramic-shatter.wav
```

```bash
swift run mere.run sfx ae encode ./branch-snap.wav -o ./branch-snap-latents.npy
swift run mere.run sfx ae decode ./branch-snap-latents.npy -o ./branch-snap-roundtrip.wav
```

```bash
swift run mere.run sfx condition text \
  "ceramic mug shattering on a tile floor" \
  -o ./ceramic-condition.safetensors
```

```bash
swift run mere.run model pull sfx-woosh-clap --accept-model-license
swift run mere.run sfx clap score \
  "ceramic mug shattering on a tile floor" \
  ./ceramic-shatter.wav
```

```bash
swift run mere.run model pull sfx-woosh-dvflow-8s --accept-model-license
swift run mere.run model pull sfx-woosh-synchformer --accept-model-license
swift run mere.run sfx video generate \
  "footsteps echoing down a concrete hallway" \
  ./silent-hallway.mp4 \
  --model sfx-woosh-dvflow-8s \
  --duration 8 \
  -o ./hallway-footsteps.wav
```

MMAudio provides a second, native 44.1 kHz path for both text-to-audio and
video-to-audio generation:

```bash
swift run mere.run model pull sfx-mmaudio-large-44k-v2 --accept-model-license
swift run mere.run sfx generate \
  "ocean waves striking a stone breakwater, close and detailed" \
  --negative-prompt "speech, music" \
  --model sfx-mmaudio-large-44k-v2 \
  --duration 8 \
  --steps 25 \
  --output ./breakwater.wav

swift run mere.run sfx video generate \
  "a skateboard rolling over rough pavement" \
  ./skateboard.mp4 \
  --negative-prompt "speech, music" \
  --model sfx-mmaudio-large-44k-v2 \
  --clip-batch-size 4 \
  --sync-batch-size 1 \
  --output ./skateboard.wav
```

MMAudio video generation needs the original video because it conditions on
both DFN5B CLIP frames and Synchformer features. A Synchformer-only `.npy`
input cannot supply the CLIP stream. The model produces 44.1 kHz audio and
defaults to 8 seconds, 25 Euler flow steps, and CFG 4.5.

## Runtime entry points

### CLI

- `Sources/MereRunCLI/Commands/SFXCommand.swift`
- `Sources/MereRunCLI/Commands/SFXAECommand.swift`
- `Sources/MereRunCLI/Commands/SFXCLAPCommand.swift`
- `Sources/MereRunCLI/Commands/SFXConditionCommand.swift`
- `Sources/MereRunCLI/Commands/SFXGenerateCommand.swift`
- `Sources/MereRunCLI/Commands/SFXVideoCommand.swift`

### Runtime

- `Sources/MereRunCore/Woosh/WooshGenerator.swift`
- `Sources/MereRunCore/Woosh/WooshDiT.swift`
- `Sources/MereRunCore/Woosh/WooshVideoDiT.swift`
- `Sources/MereRunCore/Woosh/WooshCLAP.swift`
- `Sources/MereRunCore/Woosh/WooshSynchformer.swift`
- `Sources/MereRunCore/Woosh/WooshRobertaTextEncoder.swift`
- `Sources/MereRunCore/Woosh/WooshVocosDecoder.swift`
- `Sources/MereRunCore/Woosh/WooshResources.swift`
- `Sources/MereRunCore/MMAudio/MMAudioGenerator.swift`
- `Sources/MereRunCore/MMAudio/MMAudioNetwork.swift`
- `Sources/MereRunCore/MMAudio/MMAudioCLIP.swift`
- `Sources/MereRunCore/MMAudio/MMAudioVAE.swift`
- `Sources/MereRunCore/MMAudio/MMAudioBigVGAN.swift`

## Reading order

1. `WooshResources.swift` for model IDs, validation, and checkpoint layout.
2. `WooshGenerator.swift` for prompt and video conditioning,
   Flow/DFlow/VFlow/DVFlow sampling, and WAV-ready samples.
3. `WooshRobertaTextEncoder.swift` for TextConditionerA and RoBERTa conditioning.
4. `WooshDiT.swift` for the native Flow and FlowMap DiTs.
5. `WooshVideoDiT.swift` for VFlow and DVFlow video-feature-conditioned DiTs.
6. `WooshSynchformer.swift` for raw-video frame preprocessing and
   Synchformer `synch_out` extraction
7. `WooshCLAP.swift` for Woosh-CLAP text and audio scoring.
8. `WooshVocosDecoder.swift` for Woosh-AE encoding and decoding.

## Contributor notes

- Woosh is a sound-effect and Foley model, not a song model. Keep it under
  `sfx`, not `music`.
- The managed install uses the Hugging Face mirror
  `AEmotionStudio/woosh-models`, which mirrors Sony Research Woosh v1.0.0.
- The native text-to-SFX public surface supports both Woosh-DFlow and
  Woosh-Flow. Woosh-DFlow is the fast default; Woosh-Flow is the original model
  and needs more steps for quality.
- TextConditionerA/TextConditionerV are exported Woosh-CLAP text-conditioning
  components. `sfx condition text` exposes the text-token conditioning tensors,
  and `sfx clap score` runs the native RoBERTa and PaSST retrieval path.
- `sfx video generate` runs native VFlow/DVFlow from a raw video file. It can
  also take precomputed Synchformer `synch_out` tensors as `.npy` with shape
  `[frames, 768]` or `[1, frames, 768]`.
- The open weights are CC-BY-NC 4.0. Generated outputs inherit the
  non-commercial restriction described by the mirror and upstream release.
- The MMAudio architecture source is MIT, while the released MMAudio
  checkpoints are separately licensed CC-BY-NC 4.0. The managed catalog and
  model-source documentation preserve that non-commercial checkpoint boundary.
- MMAudio's mounted Apple DFN5B CLIP conditioner is separately research-only
  under the Apple Machine Learning Research Model License Agreement. The
  managed model keeps Apple's exact license file under `clip/LICENSE`.
