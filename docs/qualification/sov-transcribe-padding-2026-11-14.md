# SOV projected-padding experiment — 2026-11-14

Keep the full padded audio input in ordinary transcription. Aggressive trimming
improves short-fixture latency, but fails recognition quality on the 140-clip,
seven-language evaluation. A more conservative 200-token margin also fails.
The 300-token margin misses the predeclared latency threshold. None of these
settings qualifies as a production inference improvement.

## Input contract and implementation

The [pinned upstream processor](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1/blob/505d8fae243924c0d8acdc622f7f0abab344eba3/processing_sovereignai_transcribe.py) pads each waveform
to complete 30-second windows and inserts 375 decoder audio tokens per window.
The native full-padding path already matches that contract.

The benchmark experiment leaves all waveform features, encoder computation,
and projection computation intact. It retains every complete window, then
keeps `ceil(final_real_samples * 375 / 480000) + margin` projected rows from the
last window, capped at 375. Margins 25, 12, and zero were the original screen.
A margin of 375 is an exact no-op. Dropped rows are no longer decoder inputs;
the audio end marker and following text positions shift. This is a model-input
change and has no general output-parity guarantee.

Mel features and all 375 projected rows in the real short-fixture captures are
bitwise identical between full padding and margin 25. The supplied 69-row
decoder tensor is an exact prefix of that full projection. Thus the measured
recognition regressions occur after the unchanged encoder/projector, when the
decoder input is shortened. The study does not isolate the contributions of
removed projected rows versus shifted rotary positions. Projected rows aligned
with waveform padding are not guaranteed redundant when waveform
samples are zero; the encoder is bidirectional and receives no padding mask.

## Recognition quality

The existing initial 70 and independent confirmation 70 FLEURS clips were reused,
20 per language across English, French, German, Spanish, Italian, Portuguese,
and Dutch. All 140 source-audio hashes are distinct; local WAV hashes were
checked before evaluation. Target weights were the saved Q8/group-64 Metal
pack, with DFlash, optimized attention, and rollback. Hints were omitted so the
model had to preserve the spoken language. There are 3,412 reference words.

The control makes 114 word errors, or 3.34% normalized WER. Its tokens match
all 140 outputs from the preceding Q8 qualification and all 140 repeated control
outputs in the margin-200 study. The default full-padding path has no detected
regression on these fixtures.

| Extra final-window tokens | Word errors | Normalized WER | Token matches to full | Inputs trimmed |
| --- | ---: | ---: | ---: | ---: |
| Full padding | 114 | 3.34% | 140/140 | 0/140 |
| 25 | 861 | 25.23% | 29/140 | 140/140 |
| 12 | 947 | 27.75% | 27/140 | 140/140 |
| 0 | 970 | 28.43% | 28/140 | 140/140 |
| 200 | 239 | 7.00% | 114/140 | 106/140 |

Normalization is Unicode NFKC/case folding, punctuation replaced by spaces, then
whitespace tokenization. This is the existing exploratory comparison metric,
not the publisher's benchmark normalizer. The JSON preserves clip-level errors,
per-language scores, complete texts/tokens, and paired clip-bootstrap intervals.

These changes include content omissions and language errors, beyond punctuation.
For example, margin 25 truncates the English Sintra/Lord Byron sentence after
“a glowing”. With margin 200, German clips `de_de-test-1796-5` and
`de_de-test-1990-6` become Dutch or mixed Dutch/German. The latter removes only
13 projected rows (362 retained versus 375), yet changes the output language.
This violates the no-language-flip criterion regardless of aggregate WER.

## Conservative follow-up

After the aggressive screen showed omissions, margins 125, 200, and 300 were
checked on 14 disjoint FLEURS training clips and five synthetic probes. The
training control makes seven errors; margins 25/125/200/300 make
111/46/7/7 errors respectively. All settings produced empty text for the three
silence probes and matched the reference short sentence padded to 480,000 and
480,001 samples. These probes alone did not reveal the multilingual problem.

Margin 200 was selected as the smallest expanded margin matching calibration
quality. It then saved 41.4 ms median short-fixture prefill time, passing the
20 ms latency screen. Full evaluation subsequently increased word errors from
114 to 239 and exposed language flips. This is a calibration failure to
generalize, not a qualified speedup. Full 140-clip quality testing was not
extended to margin 300 after it failed the latency screen.

## Resident performance

Release CLI on the M5 Max, 128 GiB, macOS 27.2. One prepared Q8 Metal target and
DFlash remain resident while each cycle alternates full and trimmed input;
cycle order reverses. One unmeasured warmup per condition precedes sampling.
No build or second evaluation model overlaps these performance runs.

The short fixture is 3.52 s, the long fixture is 46.57 s, and the 93.14 s fixture
repeats the long audio and crosses sliding-cache wrap. Rows report medians of
adjacent-pair reductions; positive values mean less time. Raw timings,
acceptance vectors, verification rounds, output tokens, and bootstrap intervals
are preserved. Timings from the Debug quality suite are not performance results.

| Clip | Margin | Prompt tokens | Prefill reduction | End-to-end reduction | Exact tokens in pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 375 | 404 → 404 | +0.4% | +0.1% | 6/6 |
| short | 25 | 404 → 98 | +46.0% | +25.6% | 6/6 |
| short | 200 | 404 → 273 | +15.1% | +11.3% | 6/6 |
| short | 300 | 404 → 373 | +4.2% | +2.4% | 8/8 |
| long | 200 | 779 → 779 | +1.9% | +0.8% | 4/4 |
| wrapped | 200 | 1529 → 1394 | +3.3% | +6.8% | 0/4 |
| long | 25 | 779 → 637 | +16.3% | +6.6% | 0/4 |
| wrapped | 25 | 1529 → 1219 | +18.1% | +2.7% | 0/4 |

The short no-op control is +0.13% end-to-end, with all six pairs matching tokens.
Margin 25 saves 118.0 ms median prefill and 25.6% end-to-end on that short
sentence, but the multilingual quality failure disqualifies it. On the long and
wrapped clips its output changes, so elapsed time is not a quality-equivalent
speed comparison. Margin 200 saves 41.4 ms and 11.3% on the short sentence but
also fails the broader quality gate. It retains every token on the 46.57 s
fixture. That row is another no-op control.

Margin 300 saves only 10.2 ms median prefill and gives a noisy +2.4% paired
end-to-end estimate, with a bootstrap interval spanning approximately
−1.5% to +6.0%. This misses the predeclared 20 ms prefill threshold and does not
justify a production change. Absolute times drift between processes; use the
within-process paired ratios, not cross-session absolute medians.

## Reproduction and validation

`model benchmark sov --audio-token-margin N --interleave-audio-tokens` selects
this research comparison. Add `--audio-tokens-first` to reverse the initial
order. Margins are bounded to 0–375. The ordinary `speech transcribe` command
continues to retain full padding. JSON records full/retained audio tokens,
prompt length, prefill/generation times, and acceptance lengths. Tensor exports
retain full `audio_embeddings` and add the exact `decoder_audio_embeddings`.

`scripts/validation/benchmark_sov_padding.py` drives paired CLI runs.
`SOVPaddingProfileTests.testHeldOutRecognition` collects optional real-checkpoint
recognition comparisons on one resident model, rotating condition order per
clip. The test's successful completion means collection succeeded; accuracy
qualification is a separate gate and failed here.

Focused checks: 29 passed, two optional checks skipped. These cover rounded
sample/token boundaries, complete windows, maximum recording length, invalid
margins, CLI parsing/exclusivity, frontend numerical parity, and generated
command documentation. Strict lint and whitespace checks pass. Full repository
checks remain deferred under the user's earlier instruction.

The first benchmark was blocked by the actual inference-admission disk reserve:
only about 13 GiB remained internally. The BF16 source existed both internally
and on the requested external SSD. All files in both copies were SHA-256
verified (53,315,578,416 bytes), then the internal duplicate was replaced by a
symlink to the SSD copy. Existing checkpoint paths continue to work, about
49.65 GiB was reclaimed, and internal free space rose to about 62 GiB. The
admission reserve was not bypassed. Both verification and relocation receipts
are included in the [machine-readable report](./sov-transcribe-padding-2026-11-14.json).

The report also records source/binary hashes, the pinned processor source
identity, all study manifests/plans, and complete raw measurements. This study
does not establish a hardware ceiling or rule out a separately trained shorter
audio representation. It rejects promotion of the tested trimming settings.
