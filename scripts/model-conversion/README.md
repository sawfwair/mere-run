# Native model conversion

These scripts are audited release tools. They are never invoked by a
`mere.run` inference command and never become a Python sidecar.

## MiniMax-H3 FL2VA and Ref2VA

`convert_minimax_h3_convrot.py` converts either pinned MiniMax-H3 transformer
partition from Comfy-Org's exact per-row ConvRot INT8 checkpoint into native
MLX affine 8-bit tensors with group size 64. It reads and validates each
tensor's embedded `convrot_groupsize` before restoring the unrotated weight
basis; that source rotation group is independent of MLX's output group size.
It then reproduces MLX's affine scale, bias, and uint32 packing. It rejects any
source whose filename, byte count, or SHA-256 differs from the pinned Hub
revision and writes a hashed conversion receipt beside the output.

The converter only handles transformer weights. It is release tooling for the
self-contained public Ref2VA package; users install that package with:

```bash
mere.run model pull video-minimax-h3-ref2va-mlx --accept-model-license
```

The package is `Sawfwair/MiniMax-H3-Ref2VA-MLX-8bit` at immutable Hub commit
`abb9114fe9d6e3cccc6376eee1abaf09d3f2a9fe`. Alongside the transformer it carries the pinned
Qwen3-VL conditioner, video/audio VAEs, tokenizer, `config.json`, `LICENSE`,
`NOTICE`, `MODIFICATIONS.md`, source manifest, conversion receipt, and hashes.
Eight-bit is the published Ref2VA quality floor; lower precision did not pass
the visual quality check.

```bash
uv run --script scripts/model-conversion/convert_minimax_h3_convrot.py \
  --partition ref2va \
  --source minimax_h3_ref2va_int8_convrot.safetensors \
  --output transformer.safetensors \
  --device cpu
```

The script pins its Python conversion dependencies and defaults to CPU for a
repeatable reference artifact. A different device is allowed for local use and
is recorded in the receipt, but its byte hash can differ because quantization
occurs at rounding boundaries.

The MiniMax-H3 Community License restricts territory and redistribution.
Conversion and use must occur in an allowed territory, and converted packages
must retain the upstream license, notice, modification disclosure, and safety
requirements. Python and CUDA are conversion tooling only; inference remains
native Swift/MLX.

`requantize_minimax_h3_mlx.py` produces a compact inference-only transformer
from that verified MLX 8-bit artifact. It requantizes affine linear weights to
4-bit/group-64, omits only the AdaLN and time-embedding weights already covered
by a verified `adaln_cache.safetensors`, and records the parent cache identity
in safetensors metadata. The script streams the output transactionally, emits
per-layer error statistics and SHA-256 receipts, and writes a matching config
that keeps the Qwen3-VL conditioner at INT8 while selecting INT4 for the
transformer. It never becomes part of the native inference path.

```bash
uv run --script scripts/model-conversion/requantize_minimax_h3_mlx.py \
  --source transformer.safetensors \
  --adaln-cache adaln_cache.safetensors \
  --config config.json \
  --output transformer-int4.safetensors \
  --output-config config-int4.json \
  --receipt transformer-int4.conversion.json
```

Install the emitted transformer and config together, retain the AdaLN cache and
receipt beside them, and set the local manifest precision/quantization to INT4.
Because cache-covered weights are intentionally absent, this compact artifact
uses the cache's exact released 31-point modulation curve. The runtime resamples
that curve for arbitrary valid schedule-point counts without restoring the
omitted inference-redundant weights.

`convert_minimax_h3_official_mlx.py` builds the publishable FL2VA bundle from
MiniMax's official BF16/FP32 release at one immutable revision. It downloads
and hashes every official input, verifies the MiniMax license bytes, and
self-tests MLX Q4/Q8 packing against an Apple Silicon byte-level fixture before
conversion. No converted or quantized third-party checkpoint is accepted.

The transformer core is quantized directly from official BF16 to affine
Q4/group-64 while precision-sensitive projections remain dense. Before
quantization, all 52 fused QKV matrices are deinterleaved from MiniMax's raw
per-head checkpoint rows into the official reference model's
`[all-q; all-k; all-v]` layout expected by the native runtime. A deterministic
QKV permutation fixture fails the conversion if that contract changes. The
Qwen3-VL conditioner is quantized directly from official BF16 to affine
Q8/group-64. The AdaLN cache is evaluated from the original released
projections before those inference-redundant weights are omitted. The official
video VAE is cast to FP16 and the official audio VAE has weight normalization
folded exactly as required by the native runtime.

```bash
HF_HOME=/workspace/hf-cache HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
  python3 scripts/model-conversion/convert_minimax_h3_official_mlx.py \
    --cache-dir /workspace/hf-cache \
    --conversion-location "CA-MTL-3, Canada" \
    --output /workspace/minimax-h3-sawfwair
```

The output carries `SOURCE_MANIFEST.json`, conversion metadata, the exact
upstream license and notices, and SHA-256 receipts for the publishable bundle.
Run the converter with `--plan` before provisioning storage, or with
`--self-test-only` to validate MLX's active backend before downloading weights.
Before publication, run the fail-closed structural and hash gate:

```bash
python3 scripts/model-conversion/validate_minimax_h3_official_artifact.py \
  /workspace/minimax-h3-sawfwair \
  --conversion-location "CA-MTL-3, Canada"
```

The validator rehashes every distributed file and checks the complete official
source manifest, license bytes, Q4/Q8 and QKV fixture receipts, component tensor
counts, dtypes, retained 50-layer conditioner, cache geometry, and forbidden
omitted branches.

## Gemma 4 12B MLX 4-bit

`convert_gemma4_12b_mlx.py` verifies Google's exact 23,919,549,408-byte Gemma 4
12B-it BF16 checkpoint at revision
`12ace6d648d72bd41519e140f1185f34d38c7e3d`, stages the pinned July 15 config,
generation, processor, tokenizer, and canonical chat-template metadata, and
converts it to affine MLX 4-bit with group size 64. Its PEP 723 environment pins
Python and every serialization dependency. The output includes
`MERERUN_CONVERSION.json` with source, tool, metadata, and emitted artifact
hashes. MLX-VLM currently rewrites processor and tokenizer configuration to its
local defaults during conversion, so the script restores the verified pinned
upstream bytes before final validation.

Gemma 4 12B is a unified multimodal architecture, so this converter uses
MLX-VLM's conversion path. Plain `mlx-lm.convert` rejects
`gemma4_unified` and must not be worked around by relabeling the config or
discarding the non-language tensors.

```bash
scripts/model-conversion/convert_gemma4_12b_mlx.py \
  --source "/path/to/google/gemma-4-12B-it" \
  --output "/path/to/gemma-4-12B-it-mlx-4bit"
```

The July update did not change the Google 12B weight blob. Conversion is useful
for producing an independently controlled MLX package; existing compatible
weights can receive the checksum-gated canonical template overlay without
being downloaded again.

The verified output is published at
[`Sawfwair/gemma-4-12B-it-MLX-4bit`](https://huggingface.co/Sawfwair/gemma-4-12B-it-MLX-4bit)
as `v1.0.0`. Managed pulls pin Hub commit
`0d75fd929a55a29ea3d431b43b2ab5142a6566bf`; the tag is the human-facing
release name, while the commit pin makes resolution reproducible.

The VDA-S, TripoSR, and InstantMesh converters have one frozen serialization
environment: CPython 3.11.15 with the exact packages in
`requirements-vfx.txt`. They fail closed on any other tool version and verify
the byte count and SHA-256 of every emitted weights, configuration, provenance,
and license file before publishing the staged directory. Their `--license-file`
argument is mandatory and accepts only the exact pinned upstream license bytes.

## DreamX-World 5B camera adapter

`extract_dreamx_camera_adapter.py` reads safetensors headers and downloads only
the 300 `cam_self_attn` tensors from the pinned public
`GD-ML/DreamX-World-5B-Cam` release. The tensors occupy 30 contiguous HTTP byte
ranges and produce a 4.22 GiB adapter instead of duplicating the 24.5 GB full
checkpoint. A sampled ordinary transformer weight from the DreamX release is
bit-identical to the managed Wan base after BF16 conversion, so the native
package composes that base with the learned camera branch.

```bash
python3 scripts/model-conversion/extract_dreamx_camera_adapter.py \
  --output camera_adapter.safetensors
```

The extractor pins revision
`a4379c7723f6ebd02139e2e8fd62d6ef523e86e3`, rejects ignored HTTP range
requests, validates every index/header entry, and writes the output atomically.
Python is acquisition tooling only; inference remains native Swift/MLX.

## Video Depth Anything Small

`convert_vda_small.py` accepts only the pinned relative or metric Apache-2.0
checkpoint. It verifies the upstream byte count and SHA-256 before using
PyTorch's weights-only loader, compares every key/dtype/shape to the committed
inventory, and emits deterministic `model.safetensors`, `config.json`, and
`SOURCE.json` files.

Both variants were converted twice from independent output directories with
byte-identical packages. Relative weights are 116,362,340 bytes with SHA-256
`85c583474dcafda4d417776431343afcdfdfc97952d8ec00029d3452c55a05a2`;
metric weights have the same byte count and SHA-256
`0acf1e186750abddf5ae867a3a659ed67cd0c041e4e524e698a0dcb40195c779`.
The runtime also pins each variant's exact config and `SOURCE.json`, plus the
11,356-byte upstream Apache-2.0 license. Self-attested package hashes are not
accepted.

```bash
python scripts/model-conversion/convert_vda_small.py \
  --variant relative \
  --source video_depth_anything_vits.pth \
  --output vda-small-native \
  --license-file /path/to/Video-Depth-Anything/LICENSE
```

The runtime can also consume the exact pinned upstream ZIP checkpoint through
its strict non-executing tensor archive reader. The converted package exists
for publishing, reproducible inspection, and deployment environments that
standardize on safetensors.

## TripoSR

`convert_triposr.py` accepts only `stabilityai/TripoSR`'s MIT checkpoint at
revision `5b521936b01fbe1890f6f9baed0254ab6351c04a`. It verifies the exact
1,677,246,742-byte source and SHA-256 before deserialization, uses PyTorch's
weights-only memory-mapped loader, and checks all 549 keys/dtypes/shapes against
the committed inventory. The output keeps upstream tensor names and layouts;
the native loader performs the two audited convolution transforms.

```bash
python scripts/model-conversion/convert_triposr.py \
  --source model.ckpt \
  --output triposr-native \
  --license-file /path/to/TripoSR/LICENSE
```

Python is release-tooling only. Native inference consumes either the exact
pinned checkpoint through the restricted non-executing archive reader or the
resulting deterministic safetensors package.

The committed converter was executed twice from independent output
directories; both files were byte-identical at 1,677,170,936 bytes with
SHA-256 `f72bb520b8b1a5639600ac818496f22d6ccb3b42d3942412bd1e2375ef780a2b`.
The canonical package also contains the 378-byte configuration, deterministic
855-byte `SOURCE.json`, and exact 1,080-byte upstream MIT license.

## InstantMesh Base reconstruction

`convert_instantmesh_base.py` accepts only TencentARC's pinned 1,253,574,354-
byte `instant_mesh_base.ckpt` at revision
`b785b4ecfb6636ef34a08c748f96f6a5686244d0`. It uses PyTorch's weights-only
loader offline, permits only the 455 `lrm_generator` reconstruction tensors,
and rejects source-camera or non-reconstruction keys. The Zero123++ view
generator and diffusion checkpoint are never inputs or outputs.

```bash
python scripts/model-conversion/convert_instantmesh_base.py \
  --source instant_mesh_base.ckpt \
  --output instantmesh-base-native \
  --license-file /path/to/InstantMesh/LICENSE
```

The deterministic runtime weights are 1,253,463,832 bytes with SHA-256
`2380601d17f6a817de0bf5328188ccea397af9d75c07b4b3cc476322dcca76af`.
The canonical package also contains the 486-byte configuration, deterministic
1,074-byte `SOURCE.json`, and exact 11,357-byte upstream Apache-2.0 license.
Native inference accepts only this verified safetensors package; it never
interprets the upstream Lightning/Pickle archive. The reconstruction weights
and pinned source are Apache-2.0. NVIDIA's separately licensed FlexiCubes and
renderer implementation are not ported; mere.run uses native marching
tetrahedra and makes no upstream topology-parity claim.
