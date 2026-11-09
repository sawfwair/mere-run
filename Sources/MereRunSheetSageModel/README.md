# SheetSage2 and MERT2

Native Swift/MLX computation for SheetSage2's adapter checkpoint and its pinned
MERT-v2 FullSong encoder. This target owns the mel frontend, ConvNeXt
subsampling, Conformer encoder, BART decoder, and structured event vocabulary.
Core owns checkpoint discovery, media input, score export, and orchestration.

Source reference: `m-a-p/SheetSage2` revision
`b83c3d79a6d98bcffaf8f9ed112c90aa887887bc`. The parent encoder is pinned to
`d8ba1c745e733b3908ce6ad16ebeb17ac7600a42`, as specified by that checkpoint.
No Python or model repository code executes during native inference.

Core is the production consumer of this internal target; no new package product
or public re-export is introduced. `SheetSage2RuntimeTests` is its independent
numerical test consumer. Core's dependency closure gains this target and keeps
the MLX dependencies it already had. Clean-build and incremental-build baseline
comparisons and linked binary size deltas were not measured for this extraction;
the local gate validates package policy and the combined build instead.

## Checkpoint qualification

`scripts/fixtures/export-sheetsage2-checkpoint-reference.py` loads separately
supplied checkpoints and the pinned upstream source. It exports an independent
CPU FP32 reference using every released layer and the full vocabulary, with
audio padding shortened to two seconds. It never downloads model weights.

Set `MERERUN_SHEETSAGE2_ROOT` to the adapter directory containing `mert2/`, and
`MERERUN_SHEETSAGE2_REFERENCE` to the exported `expected.safetensors`. Run:

```bash
MLX_ENABLE_TF32=0 MERERUN_TEST_MLX_DEVICE=gpu swift test \
  --filter SheetSage2CheckpointTests
```

Set `MERERUN_SHEETSAGE2_MEASUREMENTS` to a new JSON output path to retain maximum
absolute errors for the frontend, subsampling, first Conformer block, weighted
encoder projection, and cached and uncached decoder logits. This comparison
qualifies bounded trained-weight arithmetic; production 300-second windows,
whole-song stitching, transcription accuracy, and listening need separate runs.
