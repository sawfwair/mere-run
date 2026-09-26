# Speech Synthesize

## Purpose

Generate WAV speech from text with Qwen3-TTS or Breeze TTS 2. Use style mode for described voices or, with the CustomVoice checkpoint, a named speaker, and clone mode when a reference profile or audio file should guide timbre.

## Required Models

- `speech-tts-qwen3-nano` for voice design/style.
- `speech-tts-qwen3-customvoice` for named speakers (`--speaker`) and clone
  workflows.
- `speech-tts-breeze-2` for English and Chinese voice design, cloning, and
  direction. Its checkpoint and self-hosted outputs are restricted to research
  and non-commercial use.

## Install And Check

```bash
mere.run model pull speech-tts-qwen3-nano
mere.run model pull speech-tts-breeze-2 --accept-model-license
mere.run speech synthesize --help
```

## Parameters

- positional text: text to speak.
- `--output`, `-o`: required WAV path.
- `--model`, `-m`: model id or local path.
- `--voice`, `-v`: natural-language voice description. Breeze TTS 2 also uses
  it with a clone reference for voice direction. With
  `--speaker` it is an optional delivery instruction, such as "Speak slowly
  and warmly"; the default description is not sent with a speaker.
- `--speaker`: a named speaker of a CustomVoice checkpoint, style mode only.
  `speech-tts-qwen3-customvoice` has `aiden`, `dylan`, `eric`, `ono_anna`,
  `ryan`, `serena`, `sohee`, `uncle_fu`, and `vivian`, matched without case. A local CustomVoice checkpoint lists its own in `config.json`
  (`talker_config.spk_id`). An unknown name stops before the weights load and
  lists the checkpoint's speakers. Other checkpoints and clone mode ignore
  `--speaker`, with a warning.
- `--mode`: `style` or `clone`. It picks which of the other options apply; the
  CLI warns about one the mode ignores.
- `--profile`: saved clone profile id or name.
- `--ref-audio`: reference audio path for clone mode. Ignored with `--profile`.
- `--ref-text`: transcript override for reference audio.
- `--language`: language hint, default `auto`.
- `--save-profile`: save clone reference for reuse. Ignored with `--profile`.
- `--temperature`: sampling temperature, default `0.6`.
- `--stream`: stream audio chunks to the WAV writer.
- `--stream-chunk-tokens`: token interval for streaming chunks.
- `--quiet`, `-q`: suppress progress.

## Languages

The set of supported languages is defined by the TTS model, not by mere.run, and can change as models are added or updated. Pass `--language auto` (the default) to let the model detect the language, or set an explicit hint when text mixes scripts. For the authoritative, up-to-date language list, see the model card for the model you are running:

- `speech-tts-qwen3-nano`: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
- `speech-tts-qwen3-customvoice`: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
- `speech-tts-breeze-2`: https://huggingface.co/BreezeBlue/Breeze-TTS-2

Run `mere.run model list` to see which TTS models are installed locally.

## Reference Audio For Cloning

Clone mode needs a short, clean recording (one speaker, little background noise, ~10-20 seconds). If you need a passage to read aloud, a phonetically balanced one captures more sounds than casual speech. For example:

> The quick brown fox jumps over the lazy dog. She sells seashells by the seashore. Please call Stella and ask her to bring these things with her from the store: six spoons of fresh snow peas, five thick slabs of blue cheese, and a snack for her brother Bob.

Record yourself reading it, then pass the same text via `--ref-text` so the model aligns the transcript to your audio:

```bash
mere.run speech synthesize \
  "Now my cloned voice can say anything." \
  --mode clone \
  --ref-audio ./my-reference.wav \
  --ref-text "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore. Please call Stella and ask her to bring these things with her from the store: six spoons of fresh snow peas, five thick slabs of blue cheese, and a snack for her brother Bob." \
  --model speech-tts-qwen3-customvoice \
  --save-profile myvoice \
  --output ./cloned.wav
```

## Named Speakers

The CustomVoice checkpoint speaks as one of its trained speakers. Pick one with
`--speaker`, and optionally steer delivery with `--voice`:

```bash
mere.run speech synthesize \
  "Thanks for calling. How can I help?" \
  --model speech-tts-qwen3-customvoice \
  --speaker ryan \
  --output ./ryan.wav

mere.run speech synthesize \
  "I can't believe we won!" \
  --model speech-tts-qwen3-customvoice \
  --speaker vivian \
  --voice "Excited and breathless" \
  --output ./vivian.wav
```

`dylan` speaks Beijing dialect and `eric` Sichuan dialect when `--language` is
`auto` or `chinese`, as the upstream model does.

## Breeze TTS 2

The model runs through a native Swift/MLX text encoder, acoustic backbone,
depth decoder, and audio codec. It does not use a Python server. The pinned
checkpoint does not download automatically because its weights and self-hosted
outputs are restricted to research and non-commercial use. Review the license
before using `model pull --accept-model-license`.

```bash
mere.run speech synthesize "Hello from Breeze." \
  --model speech-tts-breeze-2 \
  --voice "A warm, confident voice" --output ./breeze.wav

mere.run speech synthesize "A directed line." \
  --model speech-tts-breeze-2 --mode clone \
  --ref-audio ./reference.wav --ref-text "Exact reference transcript." \
  --voice "Speak slowly and softly" --output ./directed.wav
```

Use the exact words spoken in the reference clip. Breeze TTS 2 does not have
named speakers. Its model generates 24 kHz
audio, and `--language` is a hint for the other TTS backends; Breeze uses the
text and reference directly.

## Prompting Patterns

- Put delivery style in `--voice`: age range, accent, energy, pace, emotion, microphone feel.
- Keep synthesis text clean. Expand abbreviations and spell numbers the way they should be read.
- Use `--language` when the text mixes scripts or languages.
- For clone mode, provide a clear transcript with `--ref-text` if auto-transcription could be wrong.

## Examples

```bash
mere.run speech synthesize \
  "Welcome to mere.run." \
  --voice "A calm, warm narrator with clear pronunciation and gentle pacing" \
  --output ./welcome.wav
```

```bash
mere.run speech synthesize \
  "This line uses the saved narrator profile." \
  --mode clone \
  --profile narrator \
  --model speech-tts-qwen3-customvoice \
  --output ./narrator.wav
```

## Iteration Tips

- Change one voice descriptor at a time: pace, emotion, timbre, then accent.
- Lower temperature for steadier delivery; raise slightly for more expressive reads.
- Save strong clone references as profiles so later runs are repeatable.

## Troubleshooting

- Clone mode fails: pass `--profile` or `--ref-audio`.
- Pronunciation is off: rewrite the text phonetically or add `--language`.
- Streaming produces too many chunks: increase `--stream-chunk-tokens`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/SpeechSynthesizeCommand.swift
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
- https://huggingface.co/BreezeBlue/Breeze-TTS-2
