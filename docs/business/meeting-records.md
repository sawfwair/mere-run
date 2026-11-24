# Prepare transcripts and timed captions

Use Transcript Tools to prepare searchable recordings, speaker-labeled review
copies, or subtitles for a training video. The workflow processes local audio
and can include supported files in nested folders. It doesn't identify speakers
by name or guarantee that a transcript preserves every word.

## Before you begin

Use recordings your team is authorized to process. Follow
[Set up companion workflows](./plugin-workflows.md) and check the
[speech runtime guide](../runtime/speech.md) for a compatible automatic speech
recognition (ASR) model and hardware. Install additional diarization and
redaction models when those features are enabled.

Install and check the plugin:

```bash
mere.run plugin install mere-transcript-tools --yes
mere.run plugin doctor mere-transcript-tools
```

A doctor result doesn't replace checking the model selected for your recording.
Choose a backend that supports the recording's language. Use a backend that
returns real timestamps for subtitles; Parakeet and Whistle support alignment
output.

## Transcribe one recording

Start with a short approved recording in a private output directory:

```bash
umask 077
mere.run plugin run mere-transcript-tools -- transcribe   --input ./meeting.wav --output-dir ./meeting-review
```

Redaction is enabled by default. The directory also contains raw ASR text for
local review, so don't share the complete folder as a redacted deliverable.
Check names, numbers, negation, and overlapping speech against the recording.
Use the reviewed `.redacted` outputs when they meet your requirements.

## Add speaker labels and subtitles

For a folder of recordings, request diarization and both caption formats:

```bash
mere.run plugin run mere-transcript-tools -- transcribe   --input ./recordings --output-dir ./subtitle-review   --diarize --subtitles both
```

Each recording gets a separate directory based on its relative input path,
including its extension. SRT and WebVTT cues use ASR timestamps. Diarization
writes speaker intervals and assigns each cue to the speaker with the greatest
overlap. Speaker labels describe model-assigned channels, not verified people.

Watch the captions with the media. Check timing at the start, middle, and end,
fast exchanges, speaker changes, and any sensitive passage. Use `--subtitles
srt` or `--subtitles vtt` when your editor needs one format.

## Plan and inspect a batch

Write a plan before a longer run, then execute that manifest:

```bash
mere.run plugin run mere-transcript-tools -- plan   --input ./recordings --output-dir ./planned-transcripts   --run-id transcript-001 --subtitles vtt
mere.run plugin run mere-transcript-tools -- run ./planned-transcripts/run.json
```

The manifest records inputs, native commands, and output hashes. Inspect errors
and per-recording coverage before distributing the batch. Keep the source
recording available while reviewers check uncertain passages. Automatic PII
reduction can miss sensitive content; review captions and transcript text
separately.

For audio cleanup before transcription, see
[Prepare audio for an edit](./audio-post.md). For backend flags and output
behavior, see the [Transcript Tools reference](https://plugins-docs.mere.run/plugins/transcript-tools).
