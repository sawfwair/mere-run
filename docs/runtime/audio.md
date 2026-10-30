# Audio runtime

The `audio` command group generates, edits, and restores audio on your Mac:

- `audio enhance` extends narrowband speech or bandwidth-limited general audio
  to 48 kHz with native Swift/MLX ports of AP-BWE and UniverSR.
- `audio generate` writes audio from a text prompt with the LTX 2.5 audio-only
  pipeline.
- `audio edit` generates or edits speech from a natural-language instruction
  with the experimental AuK runtime.

For stem separation, reverb removal, and noise removal, see
[`music separate`](./music.md). For sound effects, see the
[SFX runtime](./sfx.md).

## Commands

| Command | What it does |
| --- | --- |
| `mere.run audio enhance` | Reconstruct a 48 kHz mono float WAV with AP-BWE speech bandwidth extension or UniverSR general-audio super-resolution. |
| `mere.run audio generate` | Generate audio from text with the native LTX 2.5 audio-only model. |
| `mere.run audio edit` | Generate or edit speech with native AuK instruction conditioning. |

## Models

| Model | Used by |
| --- | --- |
| `audio-enhance-ap-bwe-16kto48k` | `audio enhance` (default) |
| `audio-enhance-universr-audio` | `audio enhance` |
| `video-ltx25-full-bf16` | `audio generate` (the complete LTX 2.5 checkpoint) |
| `audio-auk-base` | `audio edit` (default; experimental) |
| `audio-auk-flash` | `audio edit` (experimental) |
| `audio-auk-thinker` | `audio edit` (the required Qwen2.5-Omni-3B encoder) |

## Enhance audio bandwidth

Both enhancement runtimes run locally, verify their pinned artifacts before
loading, and keep machine-readable results on stdout.

```bash
swift run mere.run model pull audio-enhance-ap-bwe-16kto48k
swift run mere.run audio enhance ./narrowband-speech.wav \
  --output ./wideband-speech.wav

swift run mere.run model pull audio-enhance-universr-audio
swift run mere.run audio enhance ./music-12k.wav \
  --model audio-enhance-universr-audio \
  --output ./music-super-resolved.wav

# Declare the effective bandwidth when a limited source is stored at 48 kHz.
swift run mere.run audio enhance ./limited-48k.wav \
  --model audio-enhance-universr-audio \
  --input-rate 16000 --seed 42
```

The command writes the float WAV and a sibling JSON manifest. The manifest
records the source and output hashes, original and decoded audio geometry,
immutable code and artifact revisions, checkpoint and config hashes, license
identities, compute type, inference controls, chunk count, and elapsed time.
The same JSON is written to stdout; progress stays on stderr.

AP-BWE is a speech bandwidth-extension model, not a general mastering model.
UniverSR covers speech, music, and sound effects, but it's still
super-resolution rather than a complete mastering chain. A decoded, finite,
non-silent output proves that the native route ran; it doesn't by itself show
perceptual improvement for a particular recording.

### AP-BWE

AP-BWE is the default model. Input is decoded to 16 kHz mono, then upsampled
threefold with a deterministic windowed-sinc (Lanczos) filter before the
neural reconstruction stage. The pinned profile uses a 1,024-point short-time
Fourier transform (STFT), a 320-sample periodic Hann window, an 80-sample hop,
512 channels, and eight paired magnitude and phase ConvNeXt blocks. Chunked
overlap uses two-second 48 kHz windows by default.

The source implementation is pinned to
[`yxlu-0102/AP-BWE@751710f`](https://github.com/yxlu-0102/AP-BWE/tree/751710f22404c27e5bcc983248f8b856a04b8422).
The official repository states that both code and pretrained weights are MIT.
The public Hugging Face transport snapshot is pinned independently. `mere.run`
accepts its 16 to 48 kHz archive only because it's byte-identical to the
official Google Drive checkpoint.

### UniverSR

UniverSR accepts effective input rates of 8, 12, 16, or 24 kHz. It applies a
1,024-point complex STFT, conditions a 394-tensor ConvNeXt V2 U-Net on the
observed low-frequency region, and reconstructs the remaining spectrum with
flow matching. The published defaults are midpoint integration, four ordinary
differential equation (ODE) steps, classifier-free guidance 1.5, and seed 42.
Files stored at their native rate supply the effective rate automatically; a
bandwidth-limited 48 kHz file requires `--input-rate`.

UniverSR's native graph is pinned to
[`woongzip1/UniverSR@26dc21c`](https://github.com/woongzip1/UniverSR/tree/26dc21c44e11f9f19e823f02b0d4641dd5ea5af2),
whose code is MIT-licensed. Its official
[`woongzip1/universr-audio@1c32948`](https://huggingface.co/woongzip1/universr-audio/tree/1c3294844285af851b6ffa56cbde4e43cd41fc2b)
checkpoint is separately licensed CC BY 4.0. The managed profile preserves
that distinction and pins the checkpoint, source configuration, and model card
independently.

See the [AP-BWE implementation report](../architecture/audio-enhancement-ap-bwe-report.md)
for admission evidence and the
[UniverSR implementation report](../architecture/audio-enhancement-universr-report.md)
for its separate graph, artifact, and license evidence.

## Generate audio from text

`audio generate` runs the LTX 2.5 text-to-audio pipeline without constructing
or denoising the video modality. It needs the complete gated LTX 2.5
checkpoint, because it uses the transformer, Gemma 4 text contexts, audio VAE
and vocoder, and DurationHead:

```bash
mere.run model pull video-ltx25-full-bf16 --accept-model-license
mere.run audio generate "ocean surf, distant gulls, and a wooden buoy bell" \
  --output ./surf.wav
mere.run audio generate "a short cinematic impact with a deep metallic tail" \
  --duration 8 --audio-cfg-guidance-scale 8 --output ./impact.wav
```

Without `--duration` or `--num-frames`, DurationHead predicts a length within
the official 1 to 20 second range; `--auto-duration MIN MAX` changes that
range. The command also accepts repeatable `--lora PATH[=STRENGTH]`,
`--sigmas`, the audio CFG and STG guidance controls, and `--enhance-prompt`.
It writes the WAV path to stdout and progress to stderr. Run
`mere.run guide audio generate` for more examples, and see the
[LTX 2.5 upstream parity record](../ltx25-upstream-parity.md) for its scope.

## Generate or edit speech with AuK

`audio edit` runs Tencent's AuK base or Flash checkpoint with an instruction
such as "Remove background noise while preserving the speaker." Both
checkpoints require `audio-auk-thinker`, the separate Qwen2.5-Omni-3B encoder.
The runtime is an experimental native FP32 path:

```bash
mere.run model pull audio-auk-base
mere.run audio edit "Say 'Welcome to the workshop' in a calm, clear voice." \
  --duration 4 --output ./welcome.wav
mere.run audio edit "Remove background noise while preserving the speaker." \
  --audio ./input.wav --output ./clean.wav
```

Text-only generation requires `--duration`. With `--audio`, the output follows
the reference length unless you set `--duration`. The base model defaults to
32 Euler steps with guidance 2. `audio-auk-flash` always uses its fixed
four-step schedule with guidance disabled. For local checkpoints, pass
`--model-path` and `--thinker-path`.

AuK passed trained-weight parity and
[16 bounded audio checks](../benchmarks/auk-native-qualification-2026-09-18.md)
on Apple Silicon. Quantized inference, CUDA, longer outputs, and broader
listening quality remain unqualified. Run `mere.run guide --model audio-auk-base`
for memory requirements, reference limits, and the qualification scope.
