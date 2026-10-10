# Whistle speech recognition

`speech-asr-whistle` runs the original Whistle checkpoint directly in Swift/MLX.
The managed download pins Cactus-Compute/whistle revision
`ca5287601bef25af26dcf2e1b2bdc0843a7c19e5` and totals about 17 MB.

## Example to adapt

```bash
mere.run model pull speech-asr-whistle
mere.run speech transcribe ./recording.wav --model speech-asr-whistle
```

On Linux, use eager CUDA execution to avoid the pinned MLX backend's graph-cache
limit during beam decoding. Set this before initializing MLX in an embedded process too:

```bash
MLX_USE_CUDA_GRAPHS=0 mere.run speech transcribe ./recording.wav --model speech-asr-whistle
```

## Controls and variants

- Use `--backend whistle` to select the same model explicitly.
- Automatic language detection supports English, German, French, Spanish,
  Italian, Dutch, and Polish. `--language de`, for example, supplies a hint.
- `--max-tokens` lowers the per-window text budget, capped at 318.
- Files longer than 30 seconds use one-second overlapping windows.
- `--beam-size 1...8` selects beam width (default 5); 1 is greedy.
- `--decoder-depth 2...8` selects physical decoder blocks by bisection (default 8).
- Repeat `--keyword` for words or phrases to bias. Use `--no-timestamps` to skip word alignment.
- `--stream` supports file and 16 kHz PCM stdin. `speech listen --model speech-asr-whistle`
  captures a microphone. Live events carry utterance times and keep one model resident.
- A local `--model` directory needs `config.json` and `whistle.cact`. The managed
  download is about 17 MB of mixed CQ2/CQ4 weights and attachments.
- `--whistle-weights fp32` uses a local `checkpoints/whistle.safetensors` baseline.
- Translation and Core ML execution are unsupported.

## Sources and validation

The native encoder and cached decoder are checked against independent FP32 and
independently dequantized CQ graph outputs. Metal kernels have asset-free numerical
fixtures, including the portable Linux projection and sparse-row operations. Word timestamps are cross-attention estimates on an 80 ms grid. The
packed path retains FP32 activation and KV arithmetic, so the C++ engine's A8
numerics, accuracy, noise rejection, total memory, and latency claims do not apply.
Multilingual accuracy, long-form recognition, and microphone capture require broader
qualification.

- [Original checkpoint](https://huggingface.co/Cactus-Compute/whistle)
- [Native runtime guide](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/whistle.md)
