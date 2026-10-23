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
  swift test --filter Drama3InstalledCheckpointTests
```

The installed-checkpoint tests are opt-in and write artifacts outside the
repository. Full speech qualification acquires the normal shared inference
lease. Released-checkpoint speech, inference peak memory, and intelligibility
remain unqualified. The catalog's 32 GB minimum and 36 GB recommendation
remain estimates.

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
