# MediaIO

`MediaIO` is the cross-platform media boundary for the CLI runtime. It keeps
Apple framework use in Apple-specific backend files and routes Linux image,
audio, and video work through `ffmpeg`/`ffprobe` subprocesses.

The target exposes small value types (`MediaImage`, `MediaAudioBuffer`, and
`VideoFrameSequence`) plus facades for image, audio, and video operations. Core
model code should depend on these facades instead of importing AVFoundation,
CoreGraphics, ImageIO, or CoreVideo directly.

`MediaImageIO.bicubicResizedRGB` owns Pillow-compatible RGB bicubic resampling
for Clef and FalconPerception preprocessing. Its separable antialiasing filter,
fixed-point coefficients, intermediate byte rounding, and clipping are checked
against independently generated Pillow fixtures. The ordinary `resized` helper
retains its existing nearest-neighbor behavior.

`rescaledRGBCHWFloat` applies an explicit FP32 rescale factor before optional
minus-one-to-one normalization. Clef uses multiplication by its processor's
factor to retain reference rounding at BF16 boundaries; `rgbCHWFloat` keeps
its existing division arithmetic for other callers.

Linux users can override executable discovery with `MERERUN_FFMPEG` and
`MERERUN_FFPROBE`.

`MediaVideoSamplingStrategy.timestampFirstAtOrAfter` selects decoded presentation
timestamps rather than estimating frame indices from nominal FPS. Clef Omni
uses it for 2-fps sampling, including variable-rate video. Reaching the frame
cap stops decoding rather than uniformly subsampling the timeline.
