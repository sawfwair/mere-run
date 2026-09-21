# Qwen Image 2.1 native qualification

This report records bounded local qualification of `image-qwen-21` on an Apple
M4 Max with 128 GiB unified memory (Mac16,6, macOS 26.5.2). It qualifies the
recorded requests and artifacts. It does not establish general image quality,
minimum hardware requirements, hosted CI status, or a released implementation.

The implementation uses Swift and MLX for tokenization, Qwen3-VL conditioning,
reference encoding, denoising with prefix KV reuse, and RGBA decoding. Python
runs the independent reference comparisons and inspects saved artifacts; it is
not part of native inference.

## Provenance

- Original checkpoint: `Qwen/Qwen-Image-2.1` at
  `b3179ad355be050328e483a9dfdd9e60cd62adfa`.
- Diffusers reference: `8d3c30bfda9b511c00992f40cff4170a5502814d`
  ([upstream integration](https://github.com/huggingface/diffusers/pull/14804)).
- Source base: `d98fb81cdf4f56a585dea133f4667cbbcd5aee1b`; the feature changes
  were uncommitted during qualification.
- Release executable used for the case matrix: SHA256
  `93bf8d6ef442feabac79c571474dbecf71b853d4ca88acceca8bbbfcc665c288`.
- Reference environment: PyTorch 2.14.0, Transformers 5.17.0, and the pinned
  Diffusers source. CPU and MPS comparisons are identified separately below.

The checkpoint license was explicitly accepted before installation. All seven
weight files were checked against published SHA256 values. Managed installation
and every recorded case preflight reported ready.

## Numerical checks

Comparisons use common input tensors and trained checkpoint weights. They do
not compare final generated pixels between PyTorch and MLX: equal seeds do not
produce the same initial noise across those frameworks.

| Check | Result |
| --- | --- |
| Official tokenizer, seven English, Chinese, Unicode, blank, and image-slot cases | Exact token IDs |
| BF16 text and vision-conditioned encoder, CPU reference | Relative L2 3.19% and 3.50%; pass 5% |
| BF16 VAE encode and decode, CPU reference | Relative L2 0.60% and 1.80%; pass 5% |
| Transformer, full-precision computation with trained weights, CPU reference | Six checks pass 0.01%; worst 0.001716% |
| BF16 transformer, 512-pixel target grid, MPS reference | Six checks pass 5%; worst 2.42% |
| BF16 transformer, tiny four-token target, CPU reference | Five checks pass; initial image-conditioned velocity fails at 14.75% |
| Same tiny BF16 transformer probe, MPS reference | Initial image-conditioned velocity also fails at 14.38% |

The transformer checks include text-only and image-conditioned inputs, plus
cached and uncached second steps. The 512-pixel probe retains a synthetic
32-pixel reference image; it does not numerically qualify the full 1024-area
reference preprocessing path. Actual CLI editing exercises that path separately.

Qualification found and corrected Q/K RMS normalization rounding: Diffusers
casts the normalized activation to the learned scale's dtype before scaling.
An exact BF16 regression test covers that ordering. This correction did not
eliminate the tiny image-conditioned discrepancy. Tight full-precision
agreement and larger-grid BF16 agreement support numerical sensitivity as an
explanation, but the failed probe remains a failure at its original tolerance.
No blanket BF16 parity claim is made.

## Native CLI cases

The case plan is
`scripts/validation/qwen-image-21-qualification-plan.json`.
Run the corresponding script from the source checkout to retain preflights,
progress events, output hashes, timing, and image inspection metrics. The
[machine-readable evidence](/qualification/qwen-image-21-native-2026-09-20.json)
records the exact prompts, seeds, controls, per-case executable hashes, and
numerical comparisons.

All twelve invocations passed preflight, returned successful receipts, and wrote
valid RGBA PNGs at the requested dimensions. Visual outcomes are separate.

| Case | Pixels | Steps | Seconds | Peak footprint (GiB) | Visual result |
| --- | --- | --- | --- | --- | --- |
| smoke-512 | 512×512 | 8 | 26.8 | 15.6 | Pass |
| replay-512 | 512×512 | 8 | 13.7 | 15.5 | Pass |
| quality-1024 | 1024×1024 | 40 | 358.9 | 20.2 | Pass |
| rgba-1024 | 1024×1024 | 40 | 362.8 | 20.2 | Pass |
| edit-single | 512×512 | 20 | 57.0 | 17.9 | Pass |
| edit-multi | 512×512 | 20 | 78.5 | 21.0 | Pass |
| cfg-512 | 512×512 | 8 | 28.4 | 15.6 | Fail |
| memory-2048 | 2048×2048 | 2 | 100.3 | 70.3 | Execution only |
| quality-2048 | 2048×2048 | 40 | 2486.0 | 70.3 | Pass |
| aspect-768x512 | 768×512 | 8 | 27.2 | 15.5 | Pass |
| references-10 | 512×512 | 2 | 262.1 | 46.0 | Execution only |
| cfg-quality-512 | 512×512 | 40 | 168.1 | 15.6 | Pass |

Times are whole CLI invocations, including loading, conditioning, denoising,
and PNG writing. These are observations on one host with ordinary background
applications, not isolated throughput benchmarks. The documentation build and,
from approximately step 16, the repository gate ran during the 40-step 2K case.
Its elapsed time includes that shared-host contention. OS peak memory footprint
includes GPU allocations and allocator caches; maximum resident size alone
underreports this workload. Observed footprint is not a proven minimum memory
requirement. Smaller-memory Macs were not tested.

## Artifact findings and limitations

The 512-pixel same-seed replay produced byte-identical PNG and RGBA pixel
hashes. The 1024- and 2048-pixel teapots have coherent lids, curved spouts, handles,
and ceramic materials. The 40-step 2K case completed in 41.4 minutes with
70.3 GiB OS peak memory footprint on the tested host. The single-reference edit changes red to cobalt blue while retaining
the recognizable teapot and scene. The two-reference edit places the strawberry
beside the teapot on the wooden table.

The transparency case produces alpha values from 0 through 255; 67.2% of pixels
have alpha below 16. Checkerboard and alpha-plane inspection confirm a visible
isolated strawberry. Minor edge speckles and a semitransparent outer border
remain. This is working RGBA generation, not a production-quality matte claim.

The eight-step CFG 3 case executes and changes the output, but overexposes the
teapot and produces a distorted or ghosted handle. It fails visual inspection. A separate 40-step CFG 1.5 case produces a
coherent teapot without those defects and passes visual inspection.
The two-step 2K case produces a blurred silhouette and qualifies execution only.
The ten-reference boundary case uses repeated references and two steps; it does
not establish composition quality with ten distinct images.

## Reproduction and remaining scope

Follow `Sources/MereRunCore/QwenImage21/README.md` for tokenizer preparation,
trained-tensor exports, full-precision diagnostics, and larger-grid BF16 checks.
The checked-in runner invokes only the native executable for inference. Keep
reference output, failed comparisons, and individual case receipts rather than
combining them into a single all-pass status.

This qualification does not cover every prompt, multilingual rendered text,
all aspect ratios, ten distinct subjects, smaller-memory hardware, long-running
API concurrency, Studio interactive behavior, or cross-framework final-image
parity. Managed catalog and shared execution integration have local contract
coverage; that is separate from GUI acceptance or a packaged release.

The final `./scripts/check.sh` passed: 4,531 XCTest cases, 344 skips, zero
failures, plus 51 Swift Testing checks. This includes strict lint, build, CLI
help sweeps, and repository hygiene checks. GPU/checkpoint opt-in tests were
run separately as described above.

After qualification text was updated, the release executable was rebuilt and
checked again on September 21. Its SHA256 is
`6b027ae79994e9bbc35b2e3bf8b01f7d65c9407c693dd85b5159ae8975cc5ec3`.
The final 512-pixel seeded smoke output exactly matches both the PNG bytes and
decoded RGBA pixels from the case matrix. The numerical runtime was unchanged.
