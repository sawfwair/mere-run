# Music Separate

## Purpose

Split a finished mix into vocal and instrumental float WAVs with native
Swift/MLX BS-RoFormer inference. No audio is uploaded and no Python runtime is
launched.

## Install

The ViperX 1297 model and its AEmotion Studio mirror are MIT-licensed. The
managed package pins the full upstream revision, weights, source config,
model-card README, and license, so no license-acceptance flag is required.

```bash
mere.run model pull music-separate-bs-roformer-viperx-1297
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

The default overlap of `2` matches the published inference config. A higher
divisor of 352800, such as `4`, spends more compute on chunk blending:

```bash
mere.run music separate ./song.wav --overlap 4 --dtype float16
```

Use `--dtype float32` for full-precision model compute. Inputs are decoded to
44.1 kHz stereo, and outputs retain the decoded frame count.

## Sources

- https://huggingface.co/AEmotionStudio/roformer-models
- https://github.com/lucidrains/BS-RoFormer
- https://github.com/ZFTurbo/Music-Source-Separation-Training
- https://github.com/pymss-project/pymss-core
