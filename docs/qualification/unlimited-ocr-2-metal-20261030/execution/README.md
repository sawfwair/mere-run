# Unlimited-OCR-2 CLI and XCTest execution comparison

The CLI/XCTest discrepancy came from different Metal binaries. The CLI
accepted a replacement kernel library under a copied provenance stamp;
XCTest loaded the repository's pinned artifact. Both paths agree byte for byte
with verified kernels and emit `9.00`. Upstream still emits `9.000` for that
item. The earlier CLI price match wasn't a valid verified-kernel comparison.

## Cause and correction

| Artifact | SHA-256 prefix | Bytes | Actual macOS target | Stamped target |
| --- | --- | ---: | --- | --- |
| Earlier CLI | `1802f1eba2f0` | 4,716,496 | 14.0 | 15.0 |
| Repository | `64431ed48517` | 3,644,520 | 15.0 | 15.0 |
| Independent fresh build | `f2ad3eaf3f4f` | 3,644,520 | 15.0 | 15.0 |

The earlier CLI and repository sidecars contained identical source-version
fields, but those fields didn't bind the stamp to the binary. Inspecting the
actual library header exposed the target mismatch. Its original compiler
options aren't established. Full hashes and sidecars are preserved in
[`kernel-profiles.json`](./kernel-profiles.json).

The earlier check of equal library hashes happened after the build restored
test resources. Direct XCTest setup subsequently replaced its resource bundle
with the vendored library. Measuring the selected libraries during execution
exposed the difference. The JPEG-versus-canonical-PNG price contrast in the
earlier reports also mixed execution paths and didn't isolate JPEG decoding.
Exact Pillow pixel/tensor parity remains valid.

The fixes cover the artifact lifecycle:

- New shader builds record `metallib-sha256`. The build verifier and CLI check
  the compiled bytes against that digest. The shipped legacy stamp accepts
  only the known pinned binary. A copied sidecar no longer validates a
  replacement library; normal CLI bootstrap replaces it from a valid source.
- Compatibility copies compare bytes. Equal-size rebuilds replace stale
  libraries, and a concurrent copy error is accepted only if the destination
  matches. Valid flat `Resources/` layouts also refresh the colocated
  library that MLX loads first.
- XCTest installs its colocated library beside the linked test image. MLX
  resolves this directory with `dladdr`; the Xcode runner directory was the
  wrong location. Tests prefer a rebuilt product with a matching binary-bound
  stamp, otherwise the repository artifact.

No model equations or checkpoint tensors changed in this follow-up. Temporary
tensor capture code was removed from the generator.

## Causal tensor check

Pixels and vision features were already exact between the two execution
paths. The first captured difference was in decoder prefill logits:

| Captured stage | Earlier maximum difference | Verified-profile difference |
| --- | ---: | ---: |
| Normalized BF16 pixels | 0 | 0 |
| Vision page features | 0 | 0 |
| Prefill logits | 0.1875 | 0 |
| Early decode logits | Nonzero | 0 |

About 69.45% of the earlier prefill logits differed. Selecting the earlier CLI
artifact for both paths made their captures and `9.000` outputs identical,
which established the cause. That profile was rejected because its binary
didn't match its stamp. Selecting the verified repository artifact made the
captured pixels, features, prefill, early decode logits, and complete receipt
outputs identical at `9.00`.

[`tensor-comparison.json`](./tensor-comparison.json) records exactness, finite
error metrics, nonfinite-mask agreement, token counts, and capture digests.
The diagnostic patch is preserved in
[`capture-diagnostics.patch.txt`](./capture-diagnostics.patch.txt) for optional
reproduction on macOS. The large tensor captures remain outside the repository.

## Repeated receipt comparison

The final matrix runs original JPEG base, JPEG auto, and canonical PNG base
through both CLI and XCTest. It uses the original checkpoint at
`c14071b549f4fcab10909f3c4acc6a2edb366947`, Metal on an M5 Max with 128 GiB,
temperature zero, TF32 disabled, a 1,024-token cap, and no-repeat size 35/window
128. The same checkpoint path is passed explicitly to both paths.

Raw streams, native stage timings, commands, per-case kernel hashes, and
upstream comparisons are in [`current/summary.json`](./current/summary.json).
The matrix checks output identity and page completion. It separately evaluates
the unchanged frozen receipt anchors; it doesn't relax transcription criteria.

With the verified repository profile, the receipt has 24/25 exact upstream
block body/type matches. The remaining body difference is `9.00` versus
`9.000`. Geometry comparisons use 17 unique exact body/type pairs, with mean
absolute coordinate difference 0.441 and maximum six on the 0–1,000 scale.
Repeated bodies are excluded from geometry matching. These are comparisons
against upstream output, without annotated ground-truth boxes.

The shared `MODELING` spelling still misses the printed `MODELLING CLAY`
anchor in both implementations. The saved upstream token trace shows that
the native repetition constraint never bans its chosen tokens. Exact upstream
price, geometry, and BF16 numerical parity remain unproven. The unrelated
eight-case document suite wasn't repeated in this follow-up.

## Validation

The repository gate passes. Its 14 CLI artifact tests and four test-library
selection/location tests pass, including regressions for copied stamps,
same-size mutation, compatibility replacement, and flat resource layouts.
The verifier rejects the exact earlier CLI artifact and accepts the rebuilt
library with its binary-bound stamp.

An independent rebuild uses the standard script's Metal 3.2, macOS 15.0, and
`-fno-fast-math` settings with the current compiler. The asset-dependent
cross-process receipt test passes with that profile, including equality of
the kernel bytes available to both processes. All 17 owned numerical
tests pass on GPU. The fresh profile also emits `9.00`; see
[`fresh/summary.json`](./fresh/summary.json) and its raw streams.

The cross-process assertion compares actual CLI and generator output. It
doesn't substitute an upstream transcript or normalize away price differences.
The final generator contains no capture environment hook. `vendor/` and the
MLX dependency checkouts remain unchanged.

## Reproduce

Run the local gate, then provide the installed original checkpoint and frozen
inputs described in the [initial qualification](../README):

```bash
./scripts/check.sh
python3 docs/qualification/unlimited-ocr-2-metal-20261030/execution/run-comparison.py \
  --binary .build/debug/mere.run \
  --test-bundle .build/out/Products/Debug/MereRunCoreTests.xctest \
  --inputs .build/qualification/unlimited-ocr-2/inputs \
  --checkpoint /path/to/Unlimited-OCR-2 \
  --output /tmp/unlimited-ocr-comparison
```

To check a fresh compiler build, use a separate output directory:

```bash
./scripts/build_mlx_metallib.sh --output /tmp/unlimited-ocr-kernels
python3 docs/qualification/unlimited-ocr-2-metal-20261030/execution/run-fresh-profile.py \
  --products .build/out/Products/Debug \
  --kernels /tmp/unlimited-ocr-kernels \
  --inputs .build/qualification/unlimited-ocr-2/inputs \
  --checkpoint /path/to/Unlimited-OCR-2 \
  --output /tmp/unlimited-ocr-fresh
```

The fresh probe temporarily installs the rebuilt product resource and restores
it afterward. Run build and artifact-changing operations sequentially; rebuilding
resources during a comparison invalidates its artifact measurements.
