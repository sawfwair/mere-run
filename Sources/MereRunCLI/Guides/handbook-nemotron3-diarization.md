# Nemotron 3 Diarization (NVIDIA)

## Purpose

Use this guide to identify anonymous speakers and their active time ranges in a recording.
The managed model ID is `speech-diarization-nemotron3`.
The released checkpoint supports up to eight speakers and overlapping speech.
Speaker labels follow first arrival; they do not identify people by name.

## Example to adapt

Review the [OpenMDW-1.1 terms](https://openmdw.ai/license/1-1/), then install
the managed checkpoint:

```bash
mere.run model pull speech-diarization-nemotron3 --accept-model-license
mere.run speech diarize ./meeting.wav --model speech-diarization-nemotron3 \
  --output ./meeting-speakers.json
```

The CLI decodes audio to 16 kHz mono. Use `--format rttm` for RTTM output.
Use `--threshold`, `--min-duration`, and `--merge-gap` to adjust segments.
Keep the same audio timeline if you combine diarization with an ASR transcript.

## Controls and variants

The native runtime uses the released model's speaker cache and FIFO context
across chunks. `--latency offline` is the default and buffers 30.4 seconds.
Other input-buffer settings are `1.04`, `0.64`, and `0.32` seconds. They do not
include compute or audio I/O time.

```bash
mere.run speech diarize ./meeting.wav --model speech-diarization-nemotron3 \
  --latency 1.04 --format rttm --output ./meeting.rttm
```

The model supplies anonymous speaker channels. Assign names only from separate
evidence.

For live input, `mere.run speech diarize-live --latency 1.04` captures the
system microphone and emits JSON Lines speaker-activity events as chunks are
processed. Use `--list-devices` and `--device` to select a microphone, or
`--stdin` for 16 kHz mono signed 16-bit little-endian PCM. The live API route
is `POST /v1/audio/diarizations/stream` with a streaming PCM request body and
JSON Lines response. The whole-file `speech diarize` and multipart API route
return a completed timeline only after the input is read.

## Sources and validation

- [NVIDIA model card](https://huggingface.co/nvidia/Nemotron-3-Diarization)
- [Speech runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/speech.md)

The runtime verifies the pinned NeMo archive checksum before loading the
checkpoint. Compare speaker changes, returning speakers, and overlap with your
recording before assigning names from separate evidence.

This handbook describes controls and source architecture. It does not certify
accuracy on a particular recording or device.
