# YuE2 checkpoint qualification

These measurements were collected on November 7, 2026, using the debug CLI on
an Apple M5 Max with 40 GPU cores and 128 GiB of unified memory, running macOS
27.2. They qualify
the recorded cases on this machine. They do not establish minimum memory
requirements or general music quality across prompts, genres, and languages.

The native source reference is YuE2 revision
`1116bea76032bee5668e5f097294d082fa0b84f0`. Model weights were downloaded
separately after explicit license acceptance; no trained weights are included
in the repository.

| Component | Pinned checkpoint revision |
| --- | --- |
| YuE2-3B | `29b3558dd46954a0cd9021dc76d5c91864a0f1c7` |
| Standard VAE | `9a94e1d0ea9f8087e98f77fa88df4a4068104d2a` |
| Legacy VAE | `d0585a9cdbb36e8b6930046974500ad7078317ca` |
| SheetSage2 | `b83c3d79a6d98bcffaf8f9ed112c90aa887887bc` |
| MERT2 FullSong parent | `d8ba1c745e733b3908ce6ad16ebeb17ac7600a42` |

## Numerical comparisons

Independent CPU references use the pinned upstream implementations and released
weights. Native comparisons run on the GPU with `MLX_ENABLE_TF32=0`. The YuE2
reference casts the released BF16 weights to FP32 to compare complete-layer
arithmetic; production song generation retains BF16 transformer weights.

| Comparison | Largest observed absolute difference |
| --- | --- |
| YuE2 AR prefill logits | `4.89e-5` |
| YuE2 cached AR logits | `3.00e-5` |
| YuE2 uncached AR logits | `3.22e-5` |
| YuE2 acoustic velocity | `2.28e-5` |
| SheetSage2/MERT2 frontend, subsampling, first block, encoder, and decoder | `8.01e-5` |
| SheetSage2 weighted encoder projection | `1.62e-5` |

The released VAE also passed full-versus-tiled decoding with 40 latent frames
and 16-frame tile cores, using a `2e-5` absolute tolerance. The pinned tokenizer
matched the frozen multilingual, Unicode-normalization, and special-token IDs.

These are bounded trained-weight tests, using a short YuE2 prefix and six latent
frames, and two seconds of audio for SheetSage2 arithmetic. Production-window
transcription is checked separately below. Equal seeds do not promise equal
samples between MLX and PyTorch.

## Transcription and listening

The independent transcription stimulus contains 32 synthesized monophonic notes
over 16 seconds at 120 BPM. A match requires the exact MIDI pitch and an onset
within 50 milliseconds, using maximum bipartite matching. Offsets and voice
labels are not scored.

The native production path uses its normal 300-second audio padding. It found
all 32 reference notes, plus one extra note: precision `0.970`, recall `1.000`,
and F1 `0.985`. The largest matched onset error was 40 milliseconds. ABC export
completed without notation diagnostics or decoder truncation; its tempo estimate
was 122 BPM. This simple synthetic case is not a general transcription benchmark.

The entire 190-token production output matched upstream CPU generation exactly.
The initial native run took 14.57 seconds and peaked at 9.53 GiB before the
checkpoint-hashing memory fix. Subsequent cover transcription retained the same
tokens with bounded unused-buffer reclamation enabled.

The six-minute generated recording also completed whole-song transcription:
844 events across two overlapping windows, with no decoder truncation, notation
error, or notation diagnostics. Its acceptance intervals cover 0–200 seconds
and 200–359.9987 seconds. Peak process footprint was 7.10 GiB. The recorded
438.33 seconds includes substantial machine-admission waiting and is not a
transcription compute benchmark. This case checks long-window execution and
export; it has no independently annotated note-accuracy reference.

The owner listened to the eight-second supplied-score piano excerpt and accepted
it as clean and instrumental. It contained finite 48 kHz stereo samples, a peak
amplitude of `0.342`, and RMS amplitude of `0.064`. This listening acceptance
applies to that excerpt, not every generated song or cover.

## Generation, memory, and queue behavior

Fixed requests use seed `831001`, full score conditioning, and 32 midpoint
steps. The short request pins both music limits to 200 frames. The long request
contains 180 measures and 675 melody notes, with both limits pinned to 9,000
frames. These are deliberate frame-limit tests; reaching the limit is reported
as truncation rather than a natural song ending.

| Case | Recorded result |
| --- | --- |
| Saved stages versus normal generation | Identical decoded float32 samples |
| Decoder-only replay | Byte-identical WAV |
| Short low-memory synthesis | Byte-identical 200-frame latents; 10.49 GiB peak versus 13.95 GiB staged |
| Legacy VAE replay | Finite, non-silent stereo output |
| Six-minute baseline generation | 624.20 seconds; 65.86 GiB peak before bounded unused-buffer reclamation |
| Six-minute acoustic replay in low-memory mode | Byte-identical 9,000-frame latents across both context chunks; 212.55 seconds and 17.47 GiB peak |
| Six-minute end-to-end low-memory generation after buffer reclamation | Byte-identical 9,000 tokens, latents, and WAV; 817.09 seconds including admission waiting, and 56.75 GiB peak |
| Long-prefix sampling after buffer reclamation | Same first 200 tokens as the baseline; 15.53 GiB peak |
| First resident API job | 47.73 seconds |
| Warm resident API job | 26.12 seconds; identical latents |
| API queue and cancellation | Capacity enforced; queued jobs skipped; next job observed running within 0.21 seconds of active cancellation |
| Cancelled-stage recovery | Completed score remains inspectable; subsequent generation reproduces the same latents |
| Recording cover | Transcription, saved stages, and audio export completed in 36.48 seconds |

The six-minute baseline contains exactly 17,279,936 samples per channel at
48 kHz, for 359.9987 seconds of audio. All samples are finite; peak amplitude is
`0.936` and RMS amplitude is `0.068`. Acoustic generation crosses two original
context chunks. Decoder cropping accounts for the small difference from 360
seconds.

Memory is process physical footprint from `/usr/bin/time -l`, converted to GiB;
it includes Metal allocations and is not just resident memory. Acoustic replay
and prefix-only measurements are separate processes, not end-to-end memory
bounds. Timings include model loading, identity checksums, stage persistence,
and export on a shared development machine. Cold API residency does not mean
the filesystem cache was cold.

Qualification found and fixed five integration bugs: the generic validator
required image-pipeline components for SheetSage2 and the legacy VAE; fresh stage
directories could fail containment checks because of URL directory metadata;
Foundation read buffers accumulated while hashing large checkpoints;
offline export could not enumerate managed license-directory links; and the
instrumental planner omitted the upstream helper's section tags. The
score-stage footprint fell from about 6.9 GiB to 0.09 GiB after the hash fix.
Long prefill and transcription now reclaim unused MLX buffers when the observed
cache exceeds 512 MiB. Active model weights and KV state are retained.

Offline export now produces a self-contained model with all artifact sizes and
SHA-256 hashes verified, including four third-party license notices. A resident
JSONL batch generated both valid requests around malformed and duplicate-ID
lines. Each valid request reproduced the baseline acoustic latents exactly;
the invalid lines retained separate errors. The copied model executed without
depending on managed-install symlinks.

## Instrumental planning failure and upstream prompt fix

One style-only request for a sixteen-bar instrumental piano score exhausted the
default 4,096-token score budget after 264.37 seconds. The runtime rejected
instrumental conversion of the truncated score. That initial request did not
qualify. The earlier instrumental and long-song cases use complete supplied
scores.

The 4,096-token cap follows `abc.max_tokens` in the pinned upstream generation
configuration. It limits ABC text generation separately from the music-frame
budget. `music generate --abc-max-tokens` can override it within the shared
24,576-token context.

A follow-up kept the same prompt, seed, and sampling settings and doubled the
score cap to 8,192. Instrumental postprocessing was deferred so the raw notation
could be retained for inspection. The score again reached its cap, after
548.97 seconds. Its instrumental part contains 64 identical copies of one
four-bar accompaniment line. This points to repetitive score planning for this
prompt and seed; the larger budget did not produce a finished score. The raw
score and repetition counts are retained in the evidence bundle under
`planning-8192/`.

Comparing the full instrumental workflow then found a prompt difference. The
[pinned upstream helper](https://github.com/multimodal-art-projection/YuE/blob/1116bea76032bee5668e5f097294d082fa0b84f0/skills/yue2-music/instrumental/scripts/instrumental.py#L344)
gives the planner Intro, Verse, Chorus, and Outro section tags by default. The
native instrumental path had supplied empty lyrics to both planning and rendering.
The native planner now uses those same planner-only tags; rendering lyrics remain
empty.

With only that planner-input change, the same style, seed, sampling, and original
4,096-token cap produced a complete score in 1,607 tokens. Conversion, semantic
generation, acoustic synthesis, and decoding all completed. The eight-second
output is finite and non-silent. This is strong evidence that the missing
planning structure contributed to this case's repetition. It does not establish
general score-planning reliability or replace a listening review of that output.
The comparison is retained under `planning-upstream-sections/`.
The corrected automatic native instrumental path also reproduced the same ABC
and all 3,624 converted score tokens, with empty rendering lyrics. The planning
cap applies to the model's 1,607 emitted tokens before conversion. This check is
retained under `planning-native-fixed/`.

A separate check replayed the original prompt and 2,048 tokens of the failed
score through the pinned upstream model in FP32. Native GPU prefill, cached and
uncached logits, and acoustic velocity matched within `4.77e-5`. At that history,
upstream's filtered distribution also selected the native continuation token
with probability 1 and excluded the score-ending token. This checks an identical
history, not independent generation. Production BF16 kernels and MLX/PyTorch
random sampling can still diverge; equal seeds do not imply equal token choices.
The evidence is retained under `score-upstream-audit/`.

## Remaining limits

Native ABC export still requires its supported 1/32 grid and bar-aligned meter
changes. Unsupported notation retains raw JSON, MIDI, and LAB analysis. Broader
recording accuracy, long-song musical coherence, other languages, and smaller
Apple Silicon machines require separate evaluation. CUDA, XPU, PyTorch compile,
and vLLM execution are outside this native MLX qualification.

## Reproduce the checks

The repository gate (`./scripts/check.sh`) and the VitePress documentation build
passed after these changes. Released-checkpoint tests ran separately with the
local assets and reference files described below; they remain opt-in during the
normal repository gate.

`scripts/fixtures/export-yue2-checkpoint-reference.py` and
`scripts/fixtures/export-sheetsage2-checkpoint-reference.py` export independent
references from separately supplied local checkpoints and pinned upstream code.
They do not download weights. The opt-in tests are `YuE2CheckpointTests`,
`SheetSage2CheckpointTests`, and the installed-checkpoint cases in `YuE2Tests`.

`scripts/validation/qualify-yue2.py` records stage timings, memory, replay hashes,
phase-loading differences, optional transcription, legacy decoding, and a long
generation request. `scripts/validation/qualify-yue2-api.py` checks the real
resident queue, artifact downloads, and cancellation. Both require local models
and fresh output directories.

The local evidence bundle is `.build/qualification/yue2-20261107/`. It includes
effective requests, raw logs, stage receipts, numerical-error JSON, transcription
tokens, note-matching results, and audio. It is ignored by Git and contains no
externally published artifacts.
