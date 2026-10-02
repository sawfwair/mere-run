# Clef Flash full-checkpoint checks — October 2, 2026

The native Swift/MLX runtime completed text, image, video, and resized-image
decisions with the full pinned `mlx-community/clef-flash-4bit` checkpoint on
Apple Silicon. All categorical choices agreed with the configured reference.
Every input token ID and schema span matched. The largest difference between
four-decimal rounded probabilities was `0.0011`.

These checks call `ClefDecisionOperation` directly in an opt-in GPU test.
Managed installation, model discovery, CLI routing, and CLI preflight passed,
but ordinary CLI inference was refused by its existing requirement for 16 GiB
of available memory on this busy Mac. No admission settings were changed.
The runtime checks therefore do not establish successful CLI inference under
normal admission or operation on a 16 GB Mac.

The [machine-readable receipt](./receipts/clef-flash-native-2026-10-02.json)
records the requests, outputs, artifact hashes, reference versions, tested
binary hash, and validation boundaries. Four short synthetic probes establish
bounded numerical agreement, not general accuracy, exact full-model parity,
or maximum-context capacity. This report makes no native latency or memory
capacity claim.

## Results

Each probe loads a fresh native operation with the original affine 4-bit/group-64
backbone, BF16 vision tower, and BF16 joint head. Python runs only the separate
reference implementation.

| Probe | Input tokens | Choice in both implementations | Maximum rounded probability difference |
| --- | ---: | --- | ---: |
| Text incident routing, ordinal urgency, and truth | 339 | Technical team | 0.0011 |
| Solid red image, 256 × 256 | 294 | Red | 0.0003 |
| Four solid red video frames, 64 × 64 | 254 | Red | 0.0004 |
| Patterned image requiring resizing, 301 × 197 | 300 | Blue | 0.0003 |

Flash uses the shared Clef runtime with a 32-layer, 4096-wide dense Qwen3.5
backbone. Its processor configuration is identical to the pinned 27B Clef
configuration. No new model layers or media preprocessing were introduced.

## Reference and reproduction

The checkpoint revision is `6822f0f244ee9e19df76908ba3302f7fe40ceea6`.
The pinned `clef_mlx.py` SHA-256 is
`5b381f596a2507885a5a736bec7caef3200fb882f91ef2fea08a6bfa8f1877d2`.
Downloaded checkpoint files match the receipt's hashes, including the upstream
LFS SHA-256 values for both backbone shards and the joint head.

As in the [27B qualification](./clef-native-qualification-2026-10-01.md),
mlx-vlm 0.7.4 ignores the checkpoint's nested video processor settings.
The independent exporter explicitly constructs `Qwen3VLVideoProcessor` with
the checkpoint's declared settings. The pinned neural model and loader remain
unchanged.

Install Flash and retrieve the reference source from its pinned revision.
Use a separate Python environment with the reference package versions in the
receipt. Then generate independent requests, media, and expected outputs:

```sh
mere.run model pull text-decide-clef-flash-4bit
python scripts/fixtures/export-clef-flash-parity.py \
  --checkpoint '/path/to/text-decide-clef-flash-4bit' \
  --reference '/path/to/pinned/clef_mlx.py' \
  --output '/path/to/flash-parity'
```

The exporter verifies checkpoint hashes and writes four request JSON files,
PNG media, and `reference.json`. Run the native GPU comparison separately:

```sh
MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_TEST_CLEF_FLASH_ROOT='/path/to/text-decide-clef-flash-4bit' \
MERERUN_TEST_CLEF_PARITY_DIR='/path/to/flash-parity' \
swift test --filter ClefFlashCheckpointParityTests
```

The test compares exact token IDs and spans, categorical choices, probabilities,
truth values, and ordinal scores, then writes `native-runtime.json` in the parity
directory. It calls the runtime directly and does not reserve CLI admission
permits. Keep reference and native GPU processes sequential. To validate the
ordinary CLI path, rerun `text decide` with enough available admission headroom.
