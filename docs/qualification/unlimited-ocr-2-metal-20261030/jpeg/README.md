# Unlimited-OCR-2 native JPEG comparison

The later [execution follow-up](../execution/README) supersedes this stage's
CLI price and geometry results. It found that those CLI runs used a Metal
binary inconsistent with its copied provenance stamp. Exact JPEG pixel/tensor
parity remains valid; the earlier `9.000` CLI result isn't a verified-kernel
parity result.

Unlimited-OCR-2 now decodes JPEG inputs with a private, statically linked
libjpeg-turbo 3.2.0 decoder. The real receipt's decoded RGB bytes and resized,
normalized BF16 input tensor match the pinned upstream Pillow capture exactly.
The public Metal CLI produces identical output from JPEG and the canonical
Pillow-decoded PNG, including the item's `9.000` price.

## Receipt results

The checkpoint remains pinned to `c14071b549f4fcab10909f3c4acc6a2edb366947`.
Runs use the original BF16 weights on an M5 Max with 128 GiB, temperature zero,
TF32 disabled, a 1,024-token cap, and no-repeat size 35/window 128.

| Public CLI input | Seconds | Block body/type matches with upstream | Maximum coordinate delta |
| --- | ---: | ---: | ---: |
| Original JPEG, base | 35.4 | 25/25 | 2 |
| Original JPEG, auto | 38.6 | 25/25 | 2 |
| Canonical PNG, base | 33.7 | 25/25 | 2 |

All three stdout files are byte-identical and match the earlier canonical PNG
CLI output. Every request emits one page and stops before its token cap, with
no stderr diagnostic. Raw streams, commands, hashes, and measurements are in
[`cli/summary.json`](./cli/summary.json). Timings include startup and model
loading; concurrent repository checks make them unsuitable as throughput
benchmarks.

The body/type comparison counts a multiset of parsed blocks. Geometry uses only
unique exact body/type matches: 18 pairs, six exact boxes, mean absolute
coordinate difference 0.333, maximum two, on the model's 0–1,000 scale.
Repeated bodies such as `RM`, `0.00`, and `9.00` are excluded from geometry
matching. These are agreement measurements against upstream output, without
annotated ground-truth boxes.

The saved upstream receipt used repetition size zero. Its token trace shows
that size 35/window 128 never bans a selected token. Both implementations still
spell the printed `MODELLING` as `MODELING`; matching upstream doesn't establish
correct transcription.

## Pixel and codec checks

| Receipt preprocessing difference from Pillow | Previous ImageIO path | Native JPEG path |
| --- | ---: | ---: |
| Maximum RGB byte difference | 41 | 0 |
| Mean RGB byte difference | 0.6918 | 0 |
| Maximum normalized BF16 tensor difference | 0.28125 | 0 |
| Mean normalized BF16 tensor difference | 0.0043135 | 0 |

See [`pixels.json`](./pixels.json) and [`pixel-parity.log.txt`](./pixel-parity.log.txt).
The decoded RGB SHA-256 and source hashes are in [`method.json`](./method.json).
The identical-pixel probe checks the original receipt JPEG against the existing
upstream tensor capture, including native letterboxing and resampling.

The decoder uses full-resolution 8-bit samples, integer slow DCT, and fancy
chroma upsampling. RGB and grayscale produce opaque RGBA; CMYK conversion
matches Pillow's inverted JPEG sample convention. ICC transforms and EXIF
rotation aren't applied, matching ordinary Pillow loading without additional
transforms. Unsupported precision and truncated inputs return decode errors.
Other formats retain the platform decoder.

Five Swift tests pass. Seven independent Pillow 13.0.0/libjpeg-turbo 3.2.0
goldens cover RGB 4:4:4, 4:2:2, and 4:2:0, progressive 4:2:0, grayscale, CMYK,
and RGB-coded JPEGs with odd image dimensions. Additional checks exercise
malformed/truncated input, concurrent decodes, signature-based selection, and
PNG fallback. The fixture generator and original expected bytes live in
`Tests/MereRunCoreTests/Fixtures/PillowJPEG/`.

`bash Tests/NativeJPEG/check.sh` also passes 6,549 decodes under AddressSanitizer
and UBSan, including every truncated prefix of those fixtures. The C error
context owns allocations across `longjmp`; no jump crosses Swift frames.
This is a bounded fixture sweep, not an exhaustive JPEG decoder audit.

## XCTest execution difference

The full-checkpoint XCTest harness still emits `9.00`. Its JPEG and canonical
PNG outputs are byte-identical to each other, but differ from CLI output.
Those paired runs share one generator instance, use the verified source
files, and pass their page/output checks. Raw records are in `harness-pair/`;
[`harness-comparison.json`](./harness-comparison.json) retains both paths.

The new native-JPEG harness output has 24/25 exact block body/type matches,
mean coordinate difference 0.441, and maximum six among 17 unique pairs.
The earlier ImageIO harness mean was 0.75, with maximum six. The matched sets
are different from the CLI's set because the price text differs.

Repeating the original receipt base/auto anchor tests still fails the two
`MODELLING CLAY` assertions shared with upstream. Those failures remain in
[`receipt-anchor-tests.log.txt`](./receipt-anchor-tests.log.txt) and `results/`.
The paired test checks equal inputs/output and doesn't replace the original
spelling acceptance criterion.

The execution follow-up resolves the CLI/XCTest difference. Earlier hash checks
ran after the build restored test resources, so they missed the library replaced
during direct XCTest execution. The binaries loaded by the two paths differed.
With verified kernels, both paths agree and emit `9.00`. Full upstream price and
BF16 numerical parity remain unproven.

## Source, ownership, and build impact

[`Sources/CMereRunJPEG/upstream.json`](../../../../Sources/CMereRunJPEG/upstream.json)
pins upstream libjpeg-turbo commit
`c85e6b905bf237038faa936dab160ebfc5da0344`, the archive digest, and hashes of
97 unmodified upstream source/header files. SwiftPM compiles 54 selected
decoder translation units plus the C wrapper. Symbols have a private prefix;
the decoder links without a system/Homebrew JPEG dylib. Python and CMake aren't
build or runtime requirements.

`MediaIO` owns the new `CMereRunJPEG` dependency and exposes the opt-in
`decodeWithPillowJPEG` facade. Unlimited-OCR-2 calls that facade. No new public
SwiftPM product is added. Ten existing targets gain this private target in
their transitive closure; the before/after inventory is in `method.json`.
Package-policy checks pass for Darwin and both Linux manifest variants.
Third-party licenses and the IJG acknowledgment accompany the source and are
listed in `THIRD_PARTY_NOTICES.md`.

Three decoder-only clean builds take 0.726–0.922 seconds; recompiling the
wrapper and relinking takes 0.149–0.185 seconds. The standalone Debug decoder
probe is 523,000 bytes, compared with a 17,144-byte empty executable. The full
Debug CLI is 243,005,200 bytes. These measurements retain dependency/compiler
artifacts and overlap other qualification work. They don't measure a complete
Swift clean build or the historical CLI size delta. See
[`build-measurements.json`](./build-measurements.json) and `measure-build.py`.

## Reproduce

The required `./scripts/check.sh` gate and `corepack pnpm docs:build` both pass.
Codec tests, the strict receipt pixel/tensor probe, the paired-input harness
check, and the public CLI comparison also pass. The XCTest section retains the
original spelling-anchor failures. [`validation.json`](./validation.json) records
the completed checks and log digests.

From the repository root, run:

```bash
swift test --filter PillowJPEGTests
bash Tests/NativeJPEG/check.sh
python3 docs/qualification/unlimited-ocr-2-metal-20261030/jpeg/run-receipts.py \
  --binary .build/debug/mere.run \
  --inputs .build/qualification/unlimited-ocr-2/inputs
./scripts/check.sh
corepack pnpm docs:build
```

The CLI probe requires the installed original checkpoint and the frozen receipt
inputs described in the [initial qualification](../README). To regenerate the
small codec goldens, use Pillow 13.0.0 built with libjpeg-turbo 3.2.0 and run
`Tests/MereRunCoreTests/Fixtures/PillowJPEG/generate.py`. The optional upstream
capture remains external because it includes large model tensors.
