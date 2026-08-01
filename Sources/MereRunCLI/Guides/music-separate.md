# Music Separate

## Purpose

Split a finished mix into two or four float WAV stems, remove room reverb, or
remove broadband noise with native Swift/MLX RoFormer inference. No audio is
uploaded and no Python runtime is launched.

## Install

The ViperX 1297 model and its AEmotion Studio mirror are MIT-licensed. The
managed package pins the full upstream revision, weights, source config,
model-card README, and license, so no license-acceptance flag is required.

```bash
mere.run model pull music-separate-bs-roformer-viperx-1297
mere.run model pull music-separate-bs-roformer-4stem
mere.run model pull music-separate-mel-roformer-dereverb
mere.run model pull music-separate-mel-roformer-denoise
```

## Separate

```bash
mere.run music separate ./song.mp3 --output-dir ./song-stems
```

The output directory contains `vocals.wav`, `instrumental.wav`, and
`separation.json`. The same manifest is emitted on stdout for automation;
model loading and chunk progress are written to stderr. The manifest records
input/output hashes, the immutable model revision and weight hash, audio
geometry, compute type, chunk count, overlap, and elapsed time.

For drums, bass, other, and vocals:

```bash
mere.run music separate ./song.mp3 \
  --model music-separate-bs-roformer-4stem \
  --output-dir ./song-4stems
```

The default comes from the selected model: `2` for ViperX, four-stem, and
dereverb, and `4` for denoise. A higher divisor of the selected model's chunk
size spends more compute on chunk blending:

```bash
mere.run music separate ./song.wav --overlap 4 --dtype float16
```

For dereverberated audio in `noreverb.wav`:

```bash
mere.run music separate ./room.wav \
  --model music-separate-mel-roformer-dereverb \
  --output-dir ./room-restored
```

For denoised audio in `dry.wav`:

```bash
mere.run music separate ./noisy.wav \
  --model music-separate-mel-roformer-denoise \
  --output-dir ./noise-restored
```

Use `--dtype float32` for full-precision model compute. Inputs are decoded to
44.1 kHz stereo, and outputs retain the decoded frame count.

## Sources

- https://huggingface.co/AEmotionStudio/roformer-models
- https://github.com/lucidrains/BS-RoFormer
- https://github.com/ZFTurbo/Music-Source-Separation-Training
- https://github.com/pymss-project/pymss-core
