# Haru-4B

`speech-haru-4b` is a full-duplex spoken dialogue model. Haru processes
one 3,200-sample frame at 24 kHz and emits both audio and text.

## Example to adapt

```bash
mere.run model pull speech-haru-4b
mere.run speech converse --input user.wav --output reply.wav \
  --system-prompt-file character.txt --reference-audio voice.wav
```

## Controls and variants

`--live` uses the macOS microphone and output. Press Ctrl-C to stop. Use
`--speak-first` to have Haru open the conversation and `--seed` to set the
sampling seed. Supply a character card as UTF-8 and use a reference clip of a
single speaker whose voice you have permission to use.

## Sources and validation

The [original BF16 checkpoint](https://huggingface.co/higurashi-ai/Haru-4B) uses
Apache License 2.0. Run `mere.run model validate speech-haru-4b` to check that the
required checkpoint and tokenizer files are present.
