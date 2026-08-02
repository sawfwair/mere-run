# Audio Enhancement Runtime

Extend narrowband speech to a 48 kHz wideband artifact with a native Swift/MLX
port of AP-BWE. The runtime is local-only, verifies the checkpoint and licenses
before loading, and keeps machine-readable results on stdout.

## Command

| Command | What it does |
| --- | --- |
| `mere.run audio enhance` | Decode speech to 16 kHz mono and reconstruct a 48 kHz mono float WAV with AP-BWE. |

## Model

- `audio-enhance-ap-bwe-16kto48k`

The source implementation is pinned to
[`yxlu-0102/AP-BWE@751710f`](https://github.com/yxlu-0102/AP-BWE/tree/751710f22404c27e5bcc983248f8b856a04b8422).
The official repository states that both code and pretrained weights are MIT.
The public Hugging Face transport snapshot is pinned independently; mere.run
accepts its 16→48 kHz archive only because it is byte-identical to the official
Google Drive checkpoint.

## Typical workflow

```bash
swift run mere.run model pull audio-enhance-ap-bwe-16kto48k
swift run mere.run audio enhance ./narrowband-speech.wav \
  --output ./wideband-speech.wav
```

Input is decoded to 16 kHz mono, then upsampled threefold with a deterministic
windowed-sinc/Lanczos filter before the neural reconstruction stage. The pinned
profile uses a 1,024-point STFT, 320-sample periodic Hann window, 80-sample hop,
512 channels, and eight paired magnitude/phase ConvNeXt blocks. Chunked overlap
uses two-second 48 kHz windows by default.

The command writes the float WAV and a sibling JSON file. That manifest records
the source and output hashes, original and decoded audio geometry, immutable
code and artifact revisions, checkpoint/config hashes, compute type, chunk
count, overlap, and elapsed time. The same JSON is written to stdout; progress
remains on stderr.

AP-BWE is a speech bandwidth-extension model, not a general mastering model.
A decoded, finite, non-silent output proves the native route executed; it does
not by itself establish perceptual improvement for a particular recording.

See the [implementation report](../architecture/audio-enhancement-ap-bwe-report.md)
for admission evidence and graph details.
