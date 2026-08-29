# FastH3 VSA DataFree Metal qualification

This receipt records a bounded native Metal generation for the managed
`video-minimax-h3-fasth3-vsa-datafree-mlx` package. It proves that the packaged
base, adapter, AdaLN cache, four-evaluation schedule, VSA-H3 path, video decoder,
audio decoder, and MP4 muxer complete together. It does not establish general
output quality or parity with the full upstream checkpoint.

## Environment

- Date: August 29, 2026
- Hardware: Apple M4 Max with 128 GB unified memory
- Operating system: macOS 26.5.2 (25F84), arm64
- mere.run source base: `16cc6bed16974308e6b941bddd8bb05573f6689b`
- Adapter revision: `bcf40ca6f457ed66f8badf13514943e390205fca`
- Adapter SHA-256:
  `42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470`
- FastH3 AdaLN cache SHA-256:
  `d5e98c47a6a8924acbae4cfe34007049f22f56df7e9a4809f8d509320067fbe1`

The managed model root used symlinks to the exact package files. This avoided a
second 82 GB local copy but exercised the same single-root paths and automatic
embedded-adapter resolution as a completed managed pull.

## Preflight

The preflight resolved a text-to-video plan at 384 by 256 pixels with 22 frames,
24 fps, synchronized generated audio, seed 7, adapter strength 1, and five
schedule points for four denoising evaluations. It reported no diagnostics. The
generated command contained only the managed model ID and did not contain an
`--h3-adapter` argument.

## Command

```bash
MERERUN_H3_PROFILE_STEPS=1 \
MERERUN_H3_PROFILE_PHASES=1 \
mere.run video generate \
  "A red kite crosses a bright coastal sky while waves break below, with synchronized wind and surf." \
  --model video-minimax-h3-fasth3-vsa-datafree-mlx \
  --width 384 \
  --height 256 \
  --num-frames 22 \
  --seed 7 \
  --output minimax-h3-fasth3-vsa-datafree-metal-384x256x22.mp4 \
  --quiet
```

## Results

| Measurement | Observed value |
| --- | ---: |
| End-to-end wall time | 96.37 s |
| Runtime generation total | 94.478 s |
| Text encoding | 28.169 s |
| Denoising | 54.417 s |
| First evaluation, including Metal compilation | 43.511 s |
| Second evaluation | 3.104 s |
| Third evaluation | 3.099 s |
| Fourth evaluation | 3.121 s |
| Video decoding | 6.577 s |
| Audio decoding | 0.961 s |
| MLX peak memory | 42.15 GiB |
| Maximum resident set size | 55,709,581,312 bytes |
| Peak process memory footprint | 56,740,766,160 bytes |
| Process swaps reported by `/usr/bin/time -l` | 0 |

System-wide swap already existed on the host, and the run did not record a
before-and-after system swap delta. The process-level observation must not be
read as a system-wide no-swap qualification.

## Output validation

The generated MP4 is 89,328 bytes with SHA-256
`e3a86b5a18a94944620dbb3b40bad7cac3689935907c5bf48f8bee181899aabc`.
`ffprobe` reported:

- H.264 video at 384 by 256 pixels, 24 fps, 22 frames, and 0.916667 seconds.
- AAC stereo audio at 32 kHz and 0.917 seconds.
- Container duration 0.917 seconds.

Decoded audio was not silent: `volumedetect` measured -28.8 dB mean volume and
-14.6 dB maximum volume. A manually inspected middle frame showed the prompted
red subject above breaking coastal waves against a blue sky.

This single short sample proves execution and media closure only. It does not
measure prompt adherence across a dataset, temporal quality at longer durations,
audio semantic accuracy, or behavior on systems with less unified memory.
