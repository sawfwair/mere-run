# MediaIO

`MediaIO` is the cross-platform media boundary for the CLI runtime. It keeps
Apple framework use in Apple-specific backend files and routes Linux image,
audio, and video work through `ffmpeg`/`ffprobe` subprocesses.

The target exposes small value types (`MediaImage`, `MediaAudioBuffer`, and
`VideoFrameSequence`) plus facades for image, audio, and video operations. Core
model code should depend on these facades instead of importing AVFoundation,
CoreGraphics, ImageIO, or CoreVideo directly.

Linux users can override executable discovery with `MERERUN_FFMPEG` and
`MERERUN_FFPROBE`.
