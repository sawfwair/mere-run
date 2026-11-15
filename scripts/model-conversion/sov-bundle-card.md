---
license: apache-2.0
pipeline_tag: automatic-speech-recognition
base_model:
- salesforce-sovereign-intelligence/SOV-Transcribe-1
tags:
- mere-run
- mlx
- quantized
- speech
- asr
- dflash
language:
- en
- fr
- de
- es
- it
- pt
- nl
---

# SOV-Transcribe-1 for mere.run — Q8

Portable native Swift/MLX weights for [mere.run](https://github.com/sawfwair/mere-run),
converted from Salesforce's [SOV-Transcribe-1](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1).
The bundle contains the audio encoder/projector, tokenizer/configuration,
packed target decoder, and an optional unchanged DFlash checkpoint in `draft/`.
It runs locally on Apple Silicon without Python or upstream model code.

**Requires the bundle loader in [this mere.run revision](https://github.com/sawfwair/mere-run/commit/@RUNTIME_REVISION@).**
This is a mere.run native format; Transformers and generic mlx-lm loaders do
not support it. SOV supports file transcription in seven languages, with
instruction/context prompts. It does not provide streaming, word timestamps,
or translation through this runtime.

## Download and transcribe

```bash
hf download Sawfwair/SOV-Transcribe-1-mere-run-q8 \
  --local-dir /Volumes/SSD/MereRunModels/sov-q8

mere.run speech transcribe ./recording.wav \
  --model /Volumes/SSD/MereRunModels/sov-q8 \
  --draft-model /Volumes/SSD/MereRunModels/sov-q8/draft \
  --max-tokens 8192 --output ./transcript.txt
```

Omit `--draft-model` for target-only decoding. Download with `--exclude "draft/*"`
to omit the optional drafter. Selecting this model directory loads saved Q8
weights directly. No original BF16 decoder or quantization pass is required;
omit `--sov-packed-weights`. On Apple GPUs the native Q8 Metal path is selected;
the qualified M5 Max additionally selects distributed expert reduction.
Hardware capacity beyond the tested 128 GiB M5 Max has not been qualified.

## Conversion and validation

Only MoE expert matrices use affine Q8 group-64 quantization. Non-expert target
parameters, including FP32 routers, retain their upstream dtype and tensor bytes.
Encoder/projector and DFlash tensor bytes are unchanged. Native packing fuses
gate/up expert weights and maps parameter names. Quantization can change
recognition relative to the original BF16 model; exact parity claims concern
this Q8 representation and its qualified execution paths.

See `conversion.json` for pinned upstream revisions, `sov-bundle.json` for the
portable layout, and `SHA256SUMS` for all published payloads. To verify a download:

```bash
cd /Volumes/SSD/MereRunModels/sov-q8
shasum -a 256 -c SHA256SUMS
```

Startup checks configuration hashes, shard sizes/headers, and tensor
completeness. It does not reread the entire payload to hash it on each launch.
Timestamps and original source directories are not part of bundle validation.

The [mere.run qualification reports](https://github.com/sawfwair/mere-run/tree/@RUNTIME_REVISION@/docs/qualification)
record Q8 byte parity, startup memory, exact token/DFlash acceptance comparisons,
and device-specific latency measurements. Broad multilingual accuracy remains
unqualified. Benchmark improvements are workload and hardware dependent.

## License and attribution

Apache License 2.0; see `LICENSE` and `NOTICE`. Original models are by Salesforce
Sovereign Intelligence, based on Gemma 4 from Google DeepMind and Whisper from
OpenAI. This Sawfwair conversion is not an upstream-endorsed release.
