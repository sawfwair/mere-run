# SOV-Transcribe-1 (Salesforce)

`speech-asr-sov-transcribe-1` transcribes files locally with a Whisper large-v3
encoder and Gemma 4 26B A4B decoder. It uses the original BF16 weights with an
FP32 MoE router. English, French, German, Spanish, Italian, Portuguese, and
Dutch are supported. Install the checkpoints explicitly; inference does not
fetch them automatically.

## Example to adapt

```bash
mere.run model pull speech-asr-sov-transcribe-1
mere.run speech transcribe ./recording.wav --backend sov --max-tokens 8192
mere.run model pull speech-asr-sov-transcribe-1-dflash
mere.run speech transcribe ./recording.wav --backend sov \
  --instruction "Preserve names and write numbers as digits." \
  --draft-model speech-asr-sov-transcribe-1-dflash --max-tokens 8192
```

## Controls and variants

- `--model` accepts the managed ID, upstream repository ID, or local checkpoint directory.
- `--instruction` replaces the default verbatim transcription instruction.
- `--language French` provides a language-specific prompt when no instruction is supplied.
- `--draft-model` selects the DFlash companion or a compatible local draft directory.
- `--sov-packed-weights /Volumes/SSD/MereRunModels/sov-q8-g64` opts into Q8 experts. The first request streams and saves a native target pack; later requests load it directly. Q8 can change recognition output. The source checkpoint is still required for the encoder and tokenizer.
- Omitting the draft uses target-only greedy decoding. The companion cannot transcribe independently.
- `--max-tokens` defaults to 448; increase it for longer audio.
- Audio is decoded to mono 16 kHz and padded to complete 30-second windows, up to 30 minutes.
- `--provider coreml --coreml-encoder /path/to/sov-coreml` selects the converted encoder/projector. Gemma and DFlash remain on MLX.
- Translation, word timestamps, streaming, microphone listening, and API serving are unsupported.

## Sources and validation

The target's pinned revision is `505d8fae243924c0d8acdc622f7f0abab344eba3`;
the draft's is `8c0b3f3a98507d2e3dd35b1bee0f147fa3bded05`. Both are
Apache 2.0 licensed. Target weights total about 53.3 GB; the experimental
memory estimate is 64 GB minimum and 96 GB recommended.

Independent NumPy fixtures verify encoder, projector, Whisper frontend, and
mixed-mask draft arithmetic. Synthetic target verification tests cover greedy
parity, acceptance, rejection, EOS, and sliding-cache rollover. Original-checkpoint English fixture transcription and cross-provider tokens
have been verified on an M5 Max. Core ML projected tensors pass the documented
numerical tolerances. Broad transcription accuracy and peak memory remain
unqualified. See the runtime guide for conversion, resident timings, and
first-use compilation costs.

- [Original target](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1)
- [DFlash companion](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1-DFlash)
- [Native runtime guide](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/sov-transcribe.md)
