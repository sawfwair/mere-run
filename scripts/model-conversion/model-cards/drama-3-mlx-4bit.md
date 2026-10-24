---
license: other
license_name: fish-audio-community-license-1.0
license_link: LICENSE
library_name: mlx
pipeline_tag: text-to-speech
base_model: fishaudio/drama-3
base_model_relation: quantized
tags:
  - mere-run
  - mlx
  - text-to-speech
  - drama3
  - 4-bit
  - custom-runtime
---

# DRAMA-3 MLX expert-only 4-bit

Use this checkpoint with the native **mere.run Swift/MLX** speech runtime on
Apple Silicon. Built with Fish Audio. This is Sawfwair's derivative of
[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3), not an official
Fish Audio release.

The bundle contains the model, tokenizer, and original DAC codec. Downloading
this export avoids downloading the 41.5 GB source and repacking its experts.
The original export contains 27,408,212,020 bytes across 99 files; publication
adds provenance and qualification documents. This is a custom mere.run tensor
layout, not a generic Transformers or `mlx-lm` checkpoint.

## License

Read the bundled [Fish Audio Community License](LICENSE) before downloading or
using the weights. This derivative retains that license, its attribution, and
its use restrictions. `Notice` describes the modifications. Review the license's
revenue threshold, hosted-service terms, attribution, and voice-consent terms.
The mere.run code license does not replace the model license.

## Download and run

Use a mere.run build containing the DRAMA-3 runtime. The validated source is
[commit a68cc7a5](https://github.com/sawfwair/mere-run/tree/a68cc7a5a94b66f065b034863108caa2bbed59d2).
For a reproducible build on Apple Silicon macOS 15 or later:

```bash
git clone https://github.com/sawfwair/mere-run.git
cd mere-run
git checkout a68cc7a5a94b66f065b034863108caa2bbed59d2
swift build
```

After reviewing and accepting the model license, use the Hugging Face CLI to
download to a directory with at least 28 GB of free space:

```bash
hf download Sawfwair/DRAMA-3-MLX-4bit --local-dir /Volumes/T7/DRAMA-3-MLX-4bit
cd /Volumes/T7/DRAMA-3-MLX-4bit
shasum -a 256 -c SHA256SUMS
```

From the mere.run checkout, bind the directory and generate speech:

```bash
.build/debug/mere.run model location bind speech-tts-drama-3 \
  /Volumes/T7/DRAMA-3-MLX-4bit --accept-model-license
.build/debug/mere.run speech synthesize \
  '<|speaker:0|> Hello! The native speech engine is running on this laptop.' \
  --model speech-tts-drama-3 --seed 42 --output ./hello.wav
```

Change the directory for your storage device. Inference runs through Swift and
MLX; Python is not part of the inference path. Use the local CLI. The API server
excludes this model, including local paths. iOS inference is unsupported.

For reference conditioning, supply a reference recording and its exact transcript:

```bash
.build/debug/mere.run speech synthesize '<|speaker:0|> Welcome to the harbor.' \
  --model speech-tts-drama-3 --mode clone \
  --ref-audio ./reference.wav --ref-text 'This is the exact reference transcript.' \
  --seed 42 --output ./clone.wav
```

Use only voices you have permission to reproduce. Reference clips must be
nonempty and at most 30 seconds. Add `--stream` for causal waveform prefixes.
The CLI accepts one reference. Language is inferred from text; `--voice`, named
`--speaker` values, and classifier-free guidance are not supported.

## Quantization and provenance

- Source: `fishaudio/drama-3@dd64a85cd82e9521f7202610ede97b367fc83946`.
- Format: `drama3-experts-q4-g64-v1`, with 88 safetensor shards.
- Routed experts: original FP8 values decoded using 128 by 128 block scales,
  then packed as MLX affine 4-bit weights with groups of 64.
- Attention, routers, embeddings, and audio-tower tensors retain source BF16.
- `codec.pth` is unchanged. The native codec executes in FP32.
- No training or fine-tuning was performed.

`drama3-quantization.json` records the source and original export SHA-256 hashes.
Its original `README.md` is preserved here as `UPSTREAM_MODEL_CARD.md`.
`SHA256SUMS` covers the published files, including this replacement model card.
`MODIFICATIONS.md` records the publication changes.

## Measured scope and limits

Seven bounded English probes passed on a 36 GiB Apple Silicon system under
normal shared admission: short and paragraph batch synthesis, reference
conditioning, short and paragraph streaming, reference-conditioned streaming,
and cancellation after the first chunk. The six completed outputs had zero ASR
word errors across 166 scored words, including repeated text. The longest probe
contained 57 words and produced 23.17 seconds of audio.

Peak process physical footprint reached **32.86 GiB** for paragraph streaming.
System swap was in use. Cold first audio took **96–113 seconds**; these results
do not establish interactive latency, warm throughput, or swap-free operation.
The 32 GiB catalog minimum remains an estimate; 36 GiB is the tested system.

Streaming sample counts matched seeded batch runs, with maximum sample error
below 0.000048 against the PCM16 batch exports. Cancellation released model
graphs without publishing a partial WAV. The 750-frame generation cap remains;
reaching it fails rather than publishing a truncated final file.

The synthetic reference verifies conditioning and text content, not perceptual
voice identity. Broader languages, multi-speaker quality, maximum-length
references, Q4 quality loss against the original model, and inference on other
platforms remain unqualified. Do not infer those capabilities from the ASR result.

`DRAMA3_EXTENDED_RESULTS.json` records the seven probes at source `cac7aa6f`.
`DRAMA3_MAIN_INTEGRATION.json` records a repeated short checkpoint smoke after
integration with main, at source `e1efec52`; its WAV was byte-identical to the
corrected-sampler short baseline. The seven-probe suite was not repeated after
that integration. See the
[qualification record](https://github.com/sawfwair/mere-run/tree/a68cc7a5a94b66f065b034863108caa2bbed59d2/docs/qualification/drama3-native-20261015)
for the full evidence and boundaries.
