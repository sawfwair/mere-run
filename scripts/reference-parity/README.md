# Reference-parity harness (Q35 / Qwen-family)

## MMAudio component oracle

`export_mmaudio_trace.py` runs the pinned upstream PyTorch modules against the
same managed checkpoints used by the native runtime. It writes deterministic
CLIP, preprocessing, MMDiT, VAE, BigVGAN-v2, and Synchformer traces. The Swift
comparison suite is opt-in because the fixture and installed model are large.

```bash
python scripts/reference-parity/export_mmaudio_trace.py \
  --model-root "$HOME/Library/Application Support/MereRun/models/sfx-mmaudio-large-44k-v2" \
  --mmaudio-source /path/to/hkchengrex/MMAudio \
  --output /tmp/mmaudio-reference.safetensors

MERERUN_TEST_MMAUDIO_ROOT="$HOME/Library/Application Support/MereRun/models/sfx-mmaudio-large-44k-v2" \
MERERUN_TEST_MMAUDIO_TRACE=/tmp/mmaudio-reference.safetensors \
swift test --filter MMAudioParityTests
```

Byte-identical greedy parity gates compare this runtime against itself, so they
cannot catch a whole class of bug: a systematically wrong forward pass that is
wrong the same way on both sides of a refactor. This harness compares the
native runtime against `mlx_lm` running the **same installed checkpoint**, at
two levels:

1. **Token streams** — greedy decode of the same rendered prompt on both
   stacks; any divergence in the first tokens is a red flag.
2. **Per-layer hidden states** — the runtime dumps per-stage stats when
   `MERERUN_Q35_DEBUG_LAYER_DUMP=<path>` is set (embeddings, every decoder
   layer, final norm; float32 L2 norm + leading values of the last prompt
   position). `mlxlm_layer_dump.py` produces the matching dump from `mlx_lm`,
   and `compare_layers.py` reports the first diverging stage.

This harness found the `norm_topk_prob` default bug (2026-07-04): checkpoints
that omit the key were routed with un-renormalized top-k scores, dampening
every MoE block ~25% by mid-stack and biasing logits toward repetition.

## Setup

```bash
uv venv --python 3.14 /tmp/mlxlm-venv
uv pip install --python /tmp/mlxlm-venv/bin/python mlx-lm "transformers<5"
```

## Run

```bash
# 1. reference side (writes ref_layers.jsonl + ref_ids.txt)
/tmp/mlxlm-venv/bin/python scripts/reference-parity/mlxlm_layer_dump.py \
  "$HOME/Library/Application Support/MereRun/models/text-agent-ornith-35b-mlx" \
  "<system prompt>" "<user prompt>" ref_layers.jsonl ref_ids.txt

# 2. native side (same prompts)
MERERUN_Q35_DEBUG_LAYER_DUMP=our_layers.jsonl \
MERERUN_Q35_DEBUG_PROMPT_TOKENS=our_ids.txt \
swift run -c release mere.run text chat --model text-agent-ornith-35b-mlx \
  --thinking --temperature 0 --max-tokens 4 -s "<system prompt>" -p "<user prompt>"

# 3. compare
python3 scripts/reference-parity/compare_layers.py \
  our_layers.jsonl ref_layers.jsonl our_ids.txt ref_ids.txt
```

Interpretation: prompt ids must be identical (else the bug is in the
template/tokenizer, not numerics). Norm ratios drifting ±2% late in the stack
is normal quantized-kernel accumulation noise; a *systematic* ratio away from
1.0 that compounds layer over layer is a real defect, and the first stage
where it appears names the guilty component.

## Video Depth Anything Small

`export_vda_small_fixture.py` runs only in an isolated development/reference
environment. It verifies the official VDA source revision and the selected
relative or metric Small checkpoint SHA-256, then freezes deterministic
normalized input and raw depth arrays for the native MLX parity gate. Pass
`--variant metric` for the metrically trained checkpoint; the default is
`relative`. It is not used by runtime inference.

`export_vda_preprocess_fixture.py` separately freezes the reference OpenCV
`INTER_CUBIC` resize, multiple-of-14 sizing, and ImageNet normalization path.

## TripoSR

`export_triposr_fixture.py` freezes the official TripoSR image-tokenizer,
triplane-transformer, scene-code, and neural-field query outputs. It accepts
only the exact MIT checkpoint and requires the official source checkout at
commit `107cefdc244c39106fa830359024f6a2f1c78871`. The script uses the
weights-only memory-mapped loader; it is parity tooling, never a runtime
dependency.

```bash
python scripts/reference-parity/export_triposr_fixture.py \
  --checkpoint model.ckpt \
  --config config.yaml \
  --upstream /path/to/TripoSR \
  --output /tmp/triposr-parity \
  --device mps
```

## InstantMesh Base reconstruction

`export_instantmesh_base_fixture.py` freezes the official reconstruction-only
image encoder, camera-conditioned multi-view tokens, triplane transformer,
scene code, and fixed neural-field queries. It accepts only the exact pinned
Apache-2.0 `instant_mesh_base.ckpt` and the official source checkout at commit
`08822c52fdc399b93ea00e4fa9e596344ed52ccc`. Zero123++, diffusion, and view
generation are not imported. The fixture exporter is reference tooling only;
runtime inference loads verified safetensors in native Swift/MLX.

```bash
python scripts/reference-parity/export_instantmesh_base_fixture.py \
  --checkpoint instant_mesh_base.ckpt \
  --upstream /path/to/InstantMesh \
  --output /tmp/instantmesh-parity \
  --device mps
```

Parity covers the learned field. Native marching-tetrahedra output is tested
for deterministic valid geometry, but its topology is intentionally not
compared with NVIDIA FlexiCubes.

`export_instantmesh_preprocess_fixture.py` separately freezes the exact pinned
Torchvision tensor-resize path used before reconstruction: bicubic
`interpolation=3`, antialiasing enabled, then clamp to `[0, 1]`. Its default
257-to-320 square fixture deliberately exercises a non-320 input without
introducing non-square size semantics.

```bash
python scripts/reference-parity/export_instantmesh_preprocess_fixture.py \
  --output /tmp/instantmesh-preprocess-parity
MERERUN_TEST_INSTANTMESH_PREPROCESS=/tmp/instantmesh-preprocess-parity \
  swift test --filter InstantMeshPreprocessorTests
```

## DreamX camera conditioning

`export_dreamx_camera_fixture.py` imports the pinned DreamX trajectory and
PRoPE source files directly and freezes relative views, normalized intrinsics,
and projective Q/K/V/output transforms. `export_dreamx_camera_block_fixture.py`
adds the released block-0 camera weights and freezes a complete learned
camera-attention output. `export_dreamx_camera_transformer_fixture.py` combines
the native Wan base with the extracted camera adapter and freezes a tiny full
30-block BF16 forward pass without downloading a duplicate 24.5 GB checkpoint.

```bash
uv run --with torch --with safetensors --with numpy --with pillow \
  --with einops --with packaging \
  python scripts/reference-parity/export_dreamx_camera_fixture.py \
  --dreamx-root /path/to/DreamX-World \
  --output /tmp/dreamx-camera-fixture.json

uv run --with torch --with safetensors --with numpy --with pillow \
  --with einops --with packaging \
  python scripts/reference-parity/export_dreamx_camera_block_fixture.py \
  --dreamx-root /path/to/DreamX-World \
  --camera-weights /path/to/camera_adapter.safetensors \
  --output /tmp/dreamx-camera-block0.safetensors

uv run --with torch --with safetensors --with numpy --with scipy \
  --with diffusers --with einops --with packaging \
  python scripts/reference-parity/export_dreamx_camera_transformer_fixture.py \
  --dreamx-root /path/to/DreamX-World \
  --base-weights /path/to/model.safetensors \
  --camera-weights /path/to/camera_adapter.safetensors \
  --output /tmp/dreamx-camera-transformer.safetensors
```

Set `MERERUN_DREAMX_CAMERA_FIXTURE`, `MERERUN_DREAMX_CAMERA_WEIGHTS`, and
`MERERUN_DREAMX_CAMERA_BLOCK_FIXTURE` or
`MERERUN_DREAMX_CAMERA_TRANSFORMER_FIXTURE` when running the corresponding
Swift parity tests. These scripts are reference tooling and are never imported
by the runtime.

The lightweight AR camera parity gate uses the checked-in fixture at
`Tests/MereRunCoreTests/Fixtures/DreamX/dreamx-ar-camera.json` by default. Set
`MERERUN_DREAMX_AR_CAMERA_FIXTURE` only to test a newly exported upstream
fixture before updating the pinned reference.

## DreamX world duration and revisit evaluation

`dreamx_eval_suite.json` defines the paper-aligned 5-second, exact
63-latent/249-frame upstream-versus-native, approximately 30-second, D×3/A×3
out-and-back, translation/rotation, and rectangular-loop scenarios.
`run_dreamx_world_eval.py` runs them through a prepared native world server and
captures receipts, media, `ffprobe` results, global-pose closure, and
scene-memory telemetry.

```bash
python3 scripts/reference-parity/run_dreamx_world_eval.py \
  --base-url http://127.0.0.1:8791 \
  --source /absolute/path/to/seed.jpg \
  --prompt "preserve this playable world and its landmarks" \
  --output /tmp/dreamx-eval
```

The report does not infer visual quality from geometry. Add pinned PSNR and
SSIM scores for revisit/loop terminal frames with:

```bash
uv run --script scripts/reference-parity/score_dreamx_world_eval.py \
  --report /tmp/dreamx-eval/report.json
```

LPIPS, DINO-Sim, VPR-Sim, SP-Match, and CLIP-Video remain explicitly unscored
until their pinned learned-metric lanes evaluate the captured media.

For the `upstream_native_15s` scenario, run the released PyTorch program with
the same source, caption, seed 42, `w`/`wj`/`wl` actions, `4`/`6`/`6` weights,
63 latent frames, chunk-relative poses, 1280x704 geometry, 16 fps, and 0.3
color correction. Then build a machine-readable paired receipt:

```bash
uv run --script scripts/reference-parity/compare_dreamx_upstream_native.py \
  --native-report /tmp/dreamx-native/report.json \
  --upstream-video /tmp/dreamx-upstream/0_workspace_vesper-game-origin.mp4 \
  --upstream-input /tmp/dreamx-upstream-eval.json \
  --source /tmp/vesper-game-origin.jpg \
  --upstream-commit a1f4c6e5e45600718e5236955f2e0702e53fc275 \
  --model-revision 67487c4a61466bb7166d30b7187dd465e0ac9f6c \
  --base-model-revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --model-sha256 fba4fd99fe1955b3fd9b2fe452a8029c9a625dc42441b5a482481f877520ebef \
  --upstream-container-digest \
    nvcr.io/nvidia/pytorch@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1 \
  --upstream-chunk-relative \
  --output /tmp/dreamx-parity-receipt.json
```

The gate requires identical scenario inputs plus 249 encoded frames at
1280x704, 16 fps, and 15.5625 seconds from both stacks. It records start,
middle, and end PSNR/SSIM only as diagnostics. The receipt also preserves the
upstream source, base/model revisions, model checksum, NGC container digest,
seed, latent-frame count, fixed 1.5 trajectory speed, color correction, and
chunk-relative flag. PyTorch/
CUDA and Swift/MLX do not share a seeded random-number implementation, so
pixel identity is neither expected nor mislabeled as the parity criterion. The
checked-in 249-frame camera fixture separately gates the full composed
trajectory against upstream.
