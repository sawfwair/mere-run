# Music Transcribe

## Purpose

Convert a finished song or multitrack bounce into instrument-separated MIDI
with native MuScriptor inference.

## Install

MuScriptor weights are gated on Hugging Face and licensed CC BY-NC 4.0. Accept
the terms for the chosen repository, then configure your token and pull it:

```bash
mere.run config set hf-token "$HF_TOKEN"
mere.run model pull music-muscriptor-medium
```

`small` is the lightest checkpoint, `medium` is the default quality/speed
balance, and `large` needs substantially more unified memory.

## Transcribe

```bash
mere.run music transcribe ./song.mp3 --output ./song.mid
```

The MIDI file contains one track per detected instrument. To tell the model
which parts to expect:

```bash
mere.run music transcribe ./song.wav \
  --instruments voice,drums,electric_bass,synth_lead \
  --output ./song.mid
```

List the exact instrument group names with:

```bash
mere.run music transcribe --list-instruments
```

## Structured output

Use JSON for one event array or JSONL for a stream-friendly file. Stdout stays
machine-readable; progress remains on stderr.

```bash
mere.run music transcribe ./song.wav --format json --output ./events.json
mere.run music transcribe ./song.wav --format jsonl --output -
```

Greedy decoding is the default. `--sampling --temperature 0.8` enables sampled
decoding. `--beam-size 4` enables slower beam search and cannot be combined with
sampling. `--strict-eos` turns a chunk that reaches the generation limit into
an error instead of returning the decoded prefix.

## Sources

- https://github.com/muscriptor/muscriptor
- https://arxiv.org/abs/2607.08168
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/MusicTranscribeCommand.swift
