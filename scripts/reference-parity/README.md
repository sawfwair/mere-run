# Reference-parity harness (Q35 / Qwen-family)

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
