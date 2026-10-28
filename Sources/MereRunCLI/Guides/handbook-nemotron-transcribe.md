# Nemotron 3.5 Transcribe (NVIDIA)

`speech-asr-nemotron-35-transcribe` loads NVIDIA's original BF16 checkpoint
in native Swift/MLX for offline file transcription and speech translation to
English. The pinned weights occupy about 67.5 GB; allow 96 GB minimum and
128 GB recommended unified memory. Installation is explicit under OpenMDW-1.1.

## Example to adapt

```bash
mere.run model pull speech-asr-nemotron-35-transcribe --accept-model-license
mere.run speech transcribe ./recording.wav --model speech-asr-nemotron-35-transcribe --max-tokens 4096
mere.run speech transcribe ./recording.wav --backend nemotron --task translate --max-tokens 4096
```

## Controls and variants

- `--backend nemotron` selects this native backend explicitly.
- `--language de` supplies a transcription language hint; translation asks for English.
- `--model /absolute/checkpoint/path` accepts the original BF16 checkpoint folder.
- `--max-tokens` controls output length; the command default is 448. Longer recordings need a larger budget.
- The combined audio, prompt, and output must fit 65,536 tokens. Audio can be 20 ms through one hour.
- The released encoder recipe splits waveform input into 30-second rows, then generates one transcript.
- `--stream` is unsupported. CTC timestamps, speaker timelines, MTP/DFlash speculation,
  audio question answering, FP8, and NVFP4 variants require separate implementation and qualification.

## Sources and validation

The managed checkpoint is pinned to
`ef905575a106fdad904f3b0ce661dccad654607d`. Native arithmetic includes the
ASR Transformer, Sortformer speaker branch and fusion, and hybrid Mamba-2,
attention, and MoE decoder. Independent scalar fixtures cover the encoder and
hybrid decoder, while CPU and Metal tests cover cached decoding and recurrent
state. The original BF16 CLI passed an exact 11-word sentence and a 46.57-second
human recording with two word errors out of 129 words (1.55% WER), including
the encoder chunk boundary. Broader accuracy, translation, and one-hour
workloads remain unqualified.

- [Original checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
- [Model license](https://openmdw.ai/license/1-1/)
- [Native runtime guide](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/nemotron-transcribe.md)
