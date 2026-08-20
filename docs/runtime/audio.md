# Audio enhancement runtime

Extend narrowband speech or bandwidth-limited general audio to a 48 kHz
artifact with native Swift/MLX ports of AP-BWE and UniverSR. Both runtimes are
local-only, verify their pinned artifacts before loading, and keep
machine-readable results on stdout.

## Command

| Command | What it does |
| --- | --- |
| `mere.run audio enhance` | Reconstruct a 48 kHz mono float WAV with AP-BWE speech bandwidth extension or UniverSR general-audio super-resolution. |

## Models

- `audio-enhance-ap-bwe-16kto48k`
- `audio-enhance-universr-audio`

The source implementation is pinned to
[`yxlu-0102/AP-BWE@751710f`](https://github.com/yxlu-0102/AP-BWE/tree/751710f22404c27e5bcc983248f8b856a04b8422).
The official repository states that both code and pretrained weights are MIT.
The public Hugging Face transport snapshot is pinned independently. `mere.run`
accepts its 16 to 48 kHz archive only because it is byte-identical to the official
Google Drive checkpoint.

UniverSR's native graph is pinned to
[`woongzip1/UniverSR@26dc21c`](https://github.com/woongzip1/UniverSR/tree/26dc21c44e11f9f19e823f02b0d4641dd5ea5af2),
whose code is MIT-licensed. Its official
[`woongzip1/universr-audio@1c32948`](https://huggingface.co/woongzip1/universr-audio/tree/1c3294844285af851b6ffa56cbde4e43cd41fc2b)
checkpoint is separately licensed CC BY 4.0. The managed profile preserves
that distinction and pins the checkpoint, source configuration, and model card
independently.

## Typical workflow

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

Input is decoded to 16 kHz mono, then upsampled threefold with a deterministic
windowed-sinc/Lanczos filter before the neural reconstruction stage. The pinned
profile uses a 1,024-point short-time Fourier transform (STFT), 320-sample
periodic Hann window, 80-sample hop,
512 channels, and eight paired magnitude/phase ConvNeXt blocks. Chunked overlap
uses two-second 48 kHz windows by default.

UniverSR accepts effective input rates of 8, 12, 16, or 24 kHz. It applies a
1,024-point complex STFT, conditions a 394-tensor ConvNeXt V2 U-Net on the
observed low-frequency region, and reconstructs the remaining spectrum with
flow matching. The published defaults are midpoint integration, four ordinary
differential equation (ODE) steps, classifier-free guidance 1.5, and seed 42.
Native-rate files supply the
effective rate automatically; a bandwidth-limited 48 kHz container requires
`--input-rate`.

The command writes the float WAV and a sibling JSON file. That manifest records
the source and output hashes, original and decoded audio geometry, immutable
code and artifact revisions, checkpoint/config hashes, license identities,
compute type, inference controls, chunk count, and elapsed time. The same JSON
is written to stdout; progress remains on stderr.

AP-BWE is a speech bandwidth-extension model, not a general mastering model.
UniverSR covers speech, music, and sound effects, but is still super-resolution
rather than a complete mastering chain. A decoded, finite, non-silent output
proves the native route executed; it does not by itself establish perceptual
improvement for a particular recording.

See the [AP-BWE implementation report](../architecture/audio-enhancement-ap-bwe-report.md)
for admission evidence and the
[UniverSR implementation report](../architecture/audio-enhancement-universr-report.md) for its
separate graph, artifact, and license evidence.
