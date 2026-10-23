# Native DRAMA-3 checkpoint qualification

The released checkpoint is pinned to
`fishaudio/drama-3@dd64a85cd82e9521f7202610ede97b367fc83946`.
All 18 source files, totaling 41,502,593,827 bytes, were verified against the
pinned Hub LFS SHA-256 or Git blob digest before conversion. Built with Fish Audio.
The original checkpoint remains subject to the Fish Audio Community License.

## Q4 export

The native CLI exported all 31,525 model tensors, packing routed experts as
affine Q4/group-64 and preserving other model dtypes and the original codec.
The standalone directory contains 99 files, including 88 safetensor shards,
totaling 27,408,212,020 bytes. The export completed in 199.92 seconds with a
2,624,113,256-byte peak physical footprint reported by macOS `time -l`.
These are conversion measurements, not inference memory or throughput results.

The `drama3-quantization.json` receipt records source and output SHA-256 hashes.
Its 11 source entries match the independently verified download manifest.
The output also preserves `LICENSE` and adds the required attribution `Notice`.

External storage exposed two allocation problems: safe mapping copied entire
shards on the removable volume, and long Foundation read/hash loops retained
autoreleased buffers. The native reader now reads only headers and requested
tensors, and both tensor reads and artifact hashing drain temporary buffers
within each iteration. The completed export validates the bounded working set.

## Codec comparison

The independent reference uses the pinned SGLang-Omni DAC graph at
`2cfb3b67becb437b70751defe37e63d3af4eedb0`, evaluated with PyTorch on CPU.
Every learned state-dict key matched after excluding regenerated rotary and
causal-mask buffers. Native inference remains Swift/MLX.

- Decoding 32 deterministic acoustic frames produced 65,536 waveform samples:
  RMSE `0.00019327203335706145`, mean absolute error
  `0.00007681941497139633`, and maximum absolute error `0.003674132749438286`.
- Encoding a one-second, 220 Hz sine wave produced identical codes for all
  220 positions across ten codebooks. The native encoder pads partial input
  to the complete 2,048-sample acoustic frame, matching released `DAC.encode`.

These synthetic component probes do not establish speech quality or Q4 quality
loss. Whole-file SHA-256 verification precedes restricted codec archive parsing.

## Local contract validation

`MERERUN_SWIFT_DISABLE_INDEX_STORE=1 ./scripts/check.sh` passed 5,445 XCTest
cases with 451 skips and zero failures, plus 193 Swift Testing checks. Strict
lint, model boundaries, all three package-policy variants, CLI help, and hygiene
checks passed. `pnpm docs:build` passed. Hosted CI and inference on other
platforms were not verified in this run.

## Reproduce

```bash
mere.run model optimize /path/to/drama-3-source --output /path/to/drama-3-q4
MERERUN_TEST_DRAMA3_ROOT=/path/to/drama-3-q4 \
MERERUN_TEST_DRAMA3_EVIDENCE=/path/to/evidence \
  swift test --filter Drama3InstalledCheckpointTests/testReleasedCheckpointSpeechAndMemory
```

The installed-checkpoint tests are opt-in and write artifacts outside the
repository. Full speech qualification acquires the normal shared inference
lease. The batch result below covers one short seeded English request on a
36 GiB system. The catalog's 32 GB minimum and broader 36 GB recommendation
remain estimates; other workflows need separate qualification.

## Runtime memory follow-up

The October 16, 2026 batch speech attempt passed normal admission and reached
autoregressive generation. It peaked at 39,699,848,680 bytes of physical
footprint. System swap grew to 23,754.94 MiB and internal disk headroom fell to
7.7 GiB. The attempt was stopped below the existing 8 GiB disk floor, before
producing a completed waveform or an ASR result. The Q4 export alone does not
establish that the runtime fits a 36 GiB system.

Batch synthesis now separates three phases: encode reference speech, generate
integer acoustic codes, and decode those codes into a waveform. It releases the
text/tower checkpoint and tokenizer before final codec loading. Streaming
requires both graphs and has a separate memory boundary. Audio-tower masks
preserve the loaded embedding dtype, avoiding promotion of BF16 matrix products
to FP32. The runtime clears inactive Metal allocation buffers between frames.

The opt-in speech probe writes `native-speech-stages.json`, recording active,
cached, and peak MLX memory and process footprint at phase and frame progress
events. It verifies that batch synthesis releases the text/tower checkpoint
before codec decoding. Repeat the released-checkpoint probe through normal
shared admission before claiming an improved measured peak or successful
36 GiB qualification. The Q4 artifact and codec arithmetic remain unchanged.

The October 23, 2026 follow-up passed admission but was stopped by the disk
headroom guard after reaching 35,490,109,896 bytes of physical footprint. No
completed waveform or ASR result exists for that attempt. Inspection found
that first-use MoE fusion retained the original gate/up expert banks alongside
the fused banks. For this checkpoint, retaining both forms can add 11.25 GiB
of packed weights and quantization metadata.

The DRAMA loader now uses the shared MoE ownership-transfer path: prepare the
fused gate/up bank for one loaded layer, release its source banks, and clear
inactive buffers before loading the next layer. Prompt processing also clears
inactive buffers between backbone layers. A loader regression checks that the
source banks are released, and a synthetic BF16 probe covers both attention
families. Phase memory evidence is written after each progress event so a
controlled stop retains the completed phase observations. The memory peak of
this ownership fix is measured in the following batch probe.

## Released-checkpoint batch result

The default fused-expert runtime at commit `4be05a235d27a56c365e2b63905e7249ca8065bf`
completed the seeded English batch probe on October 23, 2026, through normal
shared admission on a system with 38,654,705,664 bytes of physical RAM (36 GiB).
The Q4 checkpoint and codec files were unchanged.

- Input: `[excited] Hello! The native speech engine is running on this laptop.`
  Temperature: `0.6`; seed: `42`.
- Output: 81 acoustic frames, 165,888 mono samples at 44,100 Hz, or
  3.761633 seconds of PCM16 audio. Sample peak: `0.494446`; clipped samples: zero.
- Wall time: 105.52 seconds, including cold checkpoint loading and waveform
  decoding. This is not a warm generation-throughput measurement.
- Peak process physical footprint: 26,165,418,296 bytes (24.37 GiB), compared
  with 39,699,848,680 bytes in the original stopped attempt. Peak MLX allocation:
  25,796,160,132 bytes. Active MLX allocation before codec decoding fell to
  20 bytes, verifying release of the text/tower checkpoint.
- Sampled total system swap rose from 3.61 GiB to 6.89 GiB. These system totals
  include other apps; this result does not establish operation without swap.
- Native Parakeet returned the exact expected ten-word transcript: zero word
  errors. Score the plain `--no-timestamps` transcript; the default export also
  repeats text in timestamped alignment lines.

This qualifies the short batch probe on the tested 36 GiB system. It does not
establish broader speech quality, Q4 quality loss against the original weights,
long requests, reference cloning, streaming capacity, or other-platform behavior.
Batch requests release resident resources; subsequent calls reload the model.

After the ownership fix, the full local gate passed 5,447 XCTest cases with
451 skips and zero failures, plus 193 Swift Testing checks. The focused runtime
and loader suite passed, and Metal tests preserved both BF16 attention paths
and expert output before and after fusion. The documentation build passed.
Hosted CI remains unverified.
