# RoFormer music source-separation implementation report

Research snapshot: 2026-08-01.

This report evaluates ViperX's BS-RoFormer checkpoint and the source-separation
tools catalogued in the community-maintained
[separation guide](https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c/edit).
The guide is a useful routing index, but model quality claims in it are not a
runtime or licensing authority. This report therefore checks implementation,
artifact, and license facts against the relevant upstream projects.

The exploration has now produced a native runtime, managed model contract, and
public `music separate` command in this worktree. This report preserves the
upstream/tool audit and records which admission and implementation claims are
proven separately from the remaining parity and quality work.

## Decision

A native Swift/MLX BS-RoFormer lane is feasible and fits the existing
`mere.run` architecture. The first parity target should be the widely mirrored
ViperX `model_bs_roformer_ep_317_sdr_12.9755.ckpt` vocals checkpoint, usually
called ViperX 1297. It is a stable, well-understood two-stem target, not a claim
that it is the best separator for every recording.

Use [`pymss`](https://github.com/pymss-project/pymss) and
[`pymss-core`](https://github.com/pymss-project/pymss-core) as frozen reference
implementations and fixture generators. Do not make either a product runtime
dependency: the product path should load safetensors and execute the graph in
the existing Swift package with MLX. This keeps inference local, avoids a
Python/Torch sidecar, and lets the CLI preserve its machine-readable-output
contract.

Checkpoint admission is closed for this exact mirror. The AEmotion Studio
repository includes an MIT `LICENSE` covering the mirrored model release, its
model card declares `license: mit` and states that the upstream model releases
use the same license, and the mirror attributes the original ViperX checkpoint
through the upstream training repository. mere.run pins the mirror revision,
license, model card, source YAML, and weights as one accepted artifact set; it
does not generalize that finding to differently hosted RoFormer checkpoints.

## Audited ViperX 1297 target

The most accessible safetensors conversion is
[`AEmotionStudio/roformer-models@d323194`](https://huggingface.co/AEmotionStudio/roformer-models/tree/d323194290f8488ea51814143806609bfbd7a1e5).
The mirror describes the source checkpoint as a vocals BS-RoFormer with SDR
12.97. That score is mirror metadata, not a result reproduced in this worktree.

| Property | Audited value |
| --- | --- |
| Architecture | BS-RoFormer |
| Training stems | `vocals`, `other` |
| Target stem | `vocals` |
| Audio | 44.1 kHz, stereo |
| Analysis window | 352,800 samples, or 8 seconds |
| STFT | 2,048-point FFT, 441-sample hop, Hann window |
| Model width/depth | 512 dimensions, 12 axial layers |
| Attention | 8 heads, 64 dimensions per head |
| Bands | 62 non-overlapping frequency bands |
| Transformer layout | One time and one frequency transformer per layer |
| Mask estimator | Depth 2, one target stem |
| Published inference defaults | Batch 4, overlap 2 |
| Converted artifact | `model.safetensors`, 639,109,056 bytes |
| LFS object ID | `fa296577206144929917601636b65ccdc407b6e6c2f209e4312d9d2b7b975a8a` |
| Safetensors inventory | 699 tensors, 159,758,796 float32 scalars |
| Mirror license | MIT, pinned `LICENSE` SHA-256 `899b277f35c41f0f5873e317319c84876f57a90d765b9e0d357035699cc4e4bc` |
| Source config | 1,881-byte YAML, SHA-256 `42e5635ceab7287b83d9591c9880966f0b61cff26a0a40a29d236bacb32e5f2c` |

The initial tensor inventory was obtained from the safetensors header without
downloading the 639 MB payload. It contains 186 band-split tensors, 264 axial
transformer tensors, 248 mask-estimator tensors, and one final-normalization
tensor. The complete artifact was subsequently downloaded through the managed
model store; its independently computed SHA-256 matches the mirror's Git LFS
object ID. The managed-model validator checks the entire downloaded artifact
again before runtime load.

The canonical architecture references are
[`lucidrains/BS-RoFormer@84f2b25`](https://github.com/lucidrains/BS-RoFormer/tree/84f2b25297424e93f721d5fb9d42291c6d934284)
and the ViperX config/checkpoint table in
[`ZFTurbo/Music-Source-Separation-Training@e247dfe`](https://github.com/ZFTTurbo/Music-Source-Separation-Training/blob/e247dfe4abc1f17c69dff719207fe045dc04413a/docs/pretrained_models.md).
Both codebases are MIT-licensed. The separately hosted AEmotion model release
also carries its own explicit MIT license and is admitted only at the pinned
revision above.

## Tool landscape

| Tool or service | What it contributes | Fit for `mere.run` |
| --- | --- | --- |
| [`pymss@7ff129f`](https://github.com/pymss-project/pymss/tree/7ff129f784bd070b60543d1715edf59284f15ebd) | Current CLI/API, Apple-Silicon MLX backend, chunking, workflows, and waveform/FFT ensembles | Primary parity oracle and UX reference; not a runtime dependency |
| [`pymss-core@9eadda9`](https://github.com/pymss-project/pymss-core/tree/9eadda9ee4bc7ad476e37e26dd769963669831c7) | Low-level BS/Mel-RoFormer graphs, MLX STFT/ISTFT, config and checkpoint helpers | Closest implementation reference for the native port |
| [`Music-Source-Separation-Training@e247dfe`](https://github.com/ZFTurbo/Music-Source-Separation-Training/tree/e247dfe4abc1f17c69dff719207fe045dc04413a) | Training/evaluation framework and a broad pretrained-model index | Fixture provenance, configs, and later training experiments |
| [`python-audio-separator@4fe3540`](https://github.com/nomadkaraoke/python-audio-separator/tree/4fe3540c249ff130bd5395c0e9377b3d16970c1a) | Mature model registry, large-file chunking, CLI, and reusable API | Behavior and packaging reference; Python/ONNX/Torch runtime is out of product scope |
| [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) | Familiar desktop workflow and broad model ecosystem | Product/UX reference only; do not embed its Python desktop stack |
| [`bs-roformer-infer@de35ada`](https://github.com/openmirlab/bs-roformer-infer/tree/de35ada5817b878da0194ee2860253dda3a9c2b2) | Registry with sources/checksums and convenient multistem defaults | Useful model-catalog and artifact-admission reference |
| [`pymss-mnn@ed6cc3b`](https://github.com/pymss-project/pymss-mnn/tree/ed6cc3b7042613370ac07411245b4030fe268e49) | Standalone C++/MNN conversion and inference without Python at runtime | Cross-platform reference; MLX remains the direct Apple-native target |
| `pymss-studio`, `pymss-ara`, and `MSST-WebUI` | Desktop, plugin, and workflow ideas | AGPL-3.0 boundary: study behavior, do not copy or embed code in this public distribution |
| MVSep and other hosted separators | Rapid comparisons across many community checkpoints | Evaluation aid only; hosted processing conflicts with the local-first runtime path |

The guide also tracks newer community checkpoint families such as PolarFormer,
HyperACE, Becruily Deux, Leap, SCNet, MDX23, DrumSep, karaoke/backing-vocal,
dereverb, denoise, and crowd-removal models. They should be treated as a
candidate backlog. Their names, quality rankings, and availability are not
sufficient artifact contracts.

ViperX 1296/1297 also appear in modern workflows as phase-correction donors.
That makes 1297 useful beyond the initial two-stem baseline, but phase
correction and ensembles should come only after one native model matches a
frozen reference end to end.

## Native integration shape

The repo already provides most of the infrastructure the port needs:

- `MediaAudioIO.decode` and `writeFloatWAV` cover platform audio decode and
  floating-point WAV output. Separation must decode directly without the
  automatic gain behavior used by some speech readers, so mixture amplitude is
  preserved.
- `ModelWeightsLoader` already reads single and sharded safetensors and converts
  tensors to the requested compute dtype.
- The pinned `mlx-swift` fork exposes `MLXFFT.rfft` and `MLXFFT.irfft`.
  `SortformerDSP` is an existing repository example of native spectral work.
- The model catalog and resolver already support the `.music` category,
  immutable revisions, checksums, and model-specific validation.
- The core contains working rotary embeddings, scaled dot-product attention,
  RMS normalization, and chunked inference patterns that can be reused without
  creating another runtime process.

A narrow first implementation now adds:

1. `MereRunCore/RoFormer/RoFormerConfiguration.swift`: typed decoding for a canonical
   JSON config. Convert the upstream YAML offline; do not parse Python YAML
   tags or use `[String: Any]` at the runtime boundary.
2. `MereRunCore/RoFormer/RoFormerDSP.swift`: centered reflect padding,
   Hann-windowed RFFT, exact inverse overlap-add normalization, and length
   restoration.
3. `MereRunCore/RoFormer/BSRoFormer.swift`: band split, alternating time and
   frequency transformer stacks with rotary attention, per-band mask
   estimators, complex mask application, and ISTFT reconstruction.
4. `MereRunCore/RoFormer/RoFormerSeparator.swift`: deterministic sequential
   chunking, overlap-add, dtype selection, and vocal/instrumental closure.
5. A `music separate` CLI command and managed-model validation kind, with tests
   beside the existing music-command and model-catalog tests.

The proposed public shape is intentionally small:

```bash
mere.run model pull music-separate-bs-roformer-viperx-1297

mere.run music separate ./mix.wav \
  --model music-separate-bs-roformer-viperx-1297 \
  --output-dir ./stems
```

The managed ID is frozen by the catalog and model resolver. The command writes
progress and diagnostics to stderr and emits a machine-readable manifest on
stdout. The output contract contains
`vocals.wav`, `instrumental.wav`, source/output hashes, model identity,
sample rate, channel count, dtype, chunk/overlap settings, and timing. The
instrumental may be derived as `original mixture - vocals`; deriving it from
the original decoded mixture is important for residual closure.

## Staged proof plan

### Stage 0: artifact and license admission — complete

- Retain the mirror's explicit MIT license and upstream attribution as part of
  the pinned artifact set.
- Download the complete frozen safetensors artifact, verify its SHA-256 and
  699-key inventory, and retain source/config/license metadata at model pull
  and runtime admission.
- Convert the upstream YAML into a strict canonical JSON config and bind its
  digest to the managed-model manifest.
- The catalog download is allowed only because these checks are now encoded in
  `RoFormerResources`.

### Stage 1: frozen parity oracle

- Pin a disposable Python environment to the audited `pymss` and `pymss-core`
  revisions; it never ships with the product.
- Generate fixtures from synthetic impulses, tones, stereo phase cases, and a
  redistributable short music mixture.
- Record input/output hashes, exact package versions, config, model digest,
  device, dtype, chunk size, and overlap.

### Stage 2: native ViperX 1297 — implemented and smoke-tested

- Prove centered-pad STFT/ISTFT round trips before loading the neural graph.
- Validate every safetensors key, dtype, and shape against a typed manifest.
- Run the real checkpoint on Apple Silicon through the managed install and
  public two-stem command.
- Compare intermediate band-split, transformer, mask, and waveform results
  with the frozen oracle before making parity or quality claims.

### Stage 3: multistem and model families

- Add a six-stem BS-RoFormer SW model after verifying its separate license and
  artifact contract.
- Evaluate four-stem ZFTurbo models, Mel-RoFormer, SCNet, and MDX families only
  through separate typed model profiles. Do not hide incompatible graphs
  behind unchecked configuration dictionaries.

### Stage 4: workflows and ensembles

- Add phase correction, waveform/FFT ensembles, karaoke/backing-vocal passes,
  dereverb, and denoise as explicit workflows whose component model identities
  and weights remain visible in the output manifest.
- Benchmark ensemble quality and cost instead of inheriting community rankings
  as product defaults.

## Acceptance gates

The native spike is complete only when all of the following are proven:

- Strict config decoding and exact 699-tensor admission reject missing, extra,
  mismatched, or non-finite values.
- STFT followed by ISTFT preserves stereo length and amplitude within a stated
  tolerance across impulses, tones, noise, and non-window-aligned lengths.
- A tiny deterministic RoFormer graph matches the frozen MLX reference at
  selected intermediate tensors and final waveform output.
- The full checkpoint matches the oracle on a redistributable fixture using
  the same dtype, chunk size, overlap, and padding semantics.
- `vocals + instrumental` closes back to the decoded mixture within a stated
  tolerance, with exact sample rate, channel count, and frame count.
- Chunk boundaries do not create discontinuities on impulse, tone, and music
  fixtures; one-shot and chunked results remain within an explicit tolerance.
- CLI stdout is machine-readable, stderr is diagnostic, failure messages are
  actionable, and the closest CLI/model-catalog tests cover parsing and
  compatibility behavior.
- `./scripts/check.sh` passes, followed by a real installed-model smoke on an
  Apple-Silicon machine.
- Wall time, peak resident memory, MLX peak memory, output hashes, device, and
  thermal context are recorded. An interrupted or thermally contaminated run
  is not performance evidence.

## Current validation boundary

The worktree now proves typed configuration decoding, the exact 699-tensor and
159,758,796-scalar native parameter inventory, all checkpoint keys and shapes,
centered stereo STFT/ISTFT round-trip behavior, published overlap/fade
planning, managed-model identity and validation, and CLI parsing. The complete
pinned checkpoint passed a real installed-model smoke on an Apple M4 Max with
128 GB unified memory and no recorded thermal or performance warning. A
one-second, 44.1 kHz stereo synthetic fixture exercised one padded eight-second
chunk in float16, wrote finite 44.1 kHz stereo float32 WAVs with 44,100 frames
each, and closed `vocals + instrumental` to the decoded source within
`1.87e-9` maximum absolute error. The manifest recorded 3.273 seconds inside
the separation path; `/usr/bin/time -l` observed 5.47 seconds wall time and
2,675,900,416 bytes maximum resident size. This is functional smoke evidence,
not a thermal-controlled performance benchmark or a music-quality result.

The remaining validation boundary is frozen-reference parity and quality:
intermediate tensor comparison, a redistributable music fixture, multiple
chunk boundaries, and the broader impulse/noise/phase corpus described above.
