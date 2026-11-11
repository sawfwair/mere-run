# Audited MLX host patches

Run `python3 scripts/prepare_mlx.py` before building from source. The repository
gate, local macOS installer, macOS app build, and CI invoke it automatically. A missing default
dependency checkout is resolved through SwiftPM first. An explicit
`--checkout` must point to an already resolved checkout of the pinned dependency.

The current patch restores affine Q8 NAX dispatch on generation-17 GPUs where
MLX reports NAX availability. It covers ordinary, non-transposed, and gathered
matmul dispatch. Other quantization modes retain the previous generation-18
restriction. Existing alignment and TF32 checks stay active.

The historical wide-output test passes on the qualified M5 Max with the current
kernel sources when Q8 acceleration is enabled. Additional checks cover real
H3 projection widths, row/column tile edges, dimensions above 32768, group sizes,
broadcasting, gathered experts (including sorted single-row batches and ragged
input/output dimensions), and CPU-referenced FP32 inputs. These tests
qualify the scoped Q8 restoration; they do not establish other modes' behavior.

## Source provenance

`scripts/patches/mlx-m5-affine-q8-nax.json` binds the patch to the Swift and core
revisions, original host-file hash, patched host-file hash, and patch hash.
Preparation accepts the pristine pin or the exact patched state. It rejects
additional tracked edits, including any shader edit, and is idempotent.

`scripts/build_mlx_metallib.sh` uses the same verifier for the known host-only
change. Shader fingerprints still come from unchanged tracked blobs. The
vendored metallib and its source stamp remain unchanged. No other dirty
dependency checkout is accepted for shader verification or generation.

After updating either MLX pin, requalify the patch before changing its manifest.
Use `--verify-only` to inspect source provenance without modifying files.

## Validation

```bash
python3 scripts/prepare_mlx.py
python3 -B -m unittest discover -s scripts/tests -p test_prepare_mlx.py
MERERUN_TEST_MLX_DEVICE=gpu MLX_ENABLE_TF32=0 \
  swift test --filter M5AffineQ8AccelerationTests
```

Repeat the Metal suite with `MLX_ENABLE_TF32=1` to exercise accelerated FP32
inputs against its independent CPU reference. Run the repository gate and
real-checkpoint qualification before publishing a dependency change. A kernel
microbenchmark alone does not establish image quality or generation latency.
