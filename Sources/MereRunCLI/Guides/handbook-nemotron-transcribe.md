# Nemotron 3.5 Transcribe (NVIDIA)

`speech-asr-nemotron-35-transcribe` defaults to published MLX 4-bit routed
experts for native offline transcription and translation to English.
The 25.09 GB download has an estimated 32 GB minimum / 48 GB recommended
memory budget. The original 67.5 GB BF16 model is
`speech-asr-nemotron-35-transcribe-bf16`, with a 96 / 128 GB budget.
Installation is explicit under OpenMDW-1.1.

## Example to adapt

```bash
mere.run model pull speech-asr-nemotron-35-transcribe --accept-model-license
mere.run speech transcribe ./recording.wav --backend nemotron --max-tokens 4096
mere.run speech transcribe ./recording.wav --backend nemotron --task translate --max-tokens 4096
mere.run model pull speech-asr-nemotron-35-transcribe-bf16 --accept-model-license
mere.run speech transcribe ./recording.wav --model speech-asr-nemotron-35-transcribe-bf16 --max-tokens 4096
```

## Controls and variants

- `--backend nemotron` selects the native backend and its INT4 default.
- `--model speech-asr-nemotron-35-transcribe-bf16` selects original BF16.
- `--language de` supplies a transcription language hint; translation asks for English.
- `--model /absolute/checkpoint/path` accepts original BF16 or a compatible local INT4 conversion.
- `--max-tokens` controls output length; the default is 448. Long recordings need a larger budget.
- Audio can be 20 ms through one hour, within 65,536 combined input/output tokens.
- The encoder splits waveform input into 30-second rows and generates one transcript.
- `--stream`, CTC timestamps, speaker timelines, audio question answering, and MTP/DFlash speculation are unsupported.

Pulling either managed variant preserves a complete legacy BF16 install under
the explicit BF16 ID before replacing the old default. INT4 downloads an
immutable published conversion; it does not convert weights during inference.
The source converter is `scripts/model-conversion/quantize_nemotron_transcribe_experts.py`.

## Sources and validation

The original NVIDIA source revision is
`ef905575a106fdad904f3b0ce661dccad654607d`. The Sawfwair conversion packs only
46 routed expert matrices to affine 4-bit/group-64; all other tensor values
and dtypes are retained. Native arithmetic includes the ASR/Speaker encoders,
fusion, Mamba-2, attention, and MoE decoder. Both installed formats passed the
exact-sentence regression; numerical fixtures exercise CPU and Metal paths.

Twelve English recordings measured 0.38% BF16 versus 0.77% INT4 WER
(2 versus 4 errors / 522 words). The 46.57-second recording's median CLI
time fell from 46.74 to 6.45 seconds; largest process footprint fell from
61.74 to 21.83 GiB on an M5 Max with 128 GiB. These small cases do not
establish broad equivalence, multilingual/translation quality, noisy speech
accuracy, one-hour performance, or minimum-memory hardware qualification.

- [Managed INT4 checkpoint](https://huggingface.co/Sawfwair/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-MLX-4bit)
- [Original BF16 checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
- [Model license](https://openmdw.ai/license/1-1/)
- [Native runtime guide](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/nemotron-transcribe.md)
