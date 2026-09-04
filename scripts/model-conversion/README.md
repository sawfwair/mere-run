# Native model conversion

## Parakeet Core ML/MLX package

`convert_parakeet_coreml.py` builds a Mere-controlled FP16 Core ML encoder and
TDT decoder for NVIDIA Parakeet TDT 0.6B v3. It retains a compact MLX decoder
as a compatibility fallback. The converter pins the NVIDIA
repository revision, source file sizes and SHA-256 values, Python packages,
Xcode version, tensor names, and static 15-second input shape. The output is
transactional and contains a complete hash closure in `parakeet-coreml.json`.

Inspect the plan without installing the conversion dependencies or downloading
the 2.5 GB checkpoint:

```bash
uv run --script scripts/model-conversion/convert_parakeet_coreml.py --plan
```

Run the conversion only on a machine with the pinned Xcode version and at
least 10 GiB of free working space:

```bash
uv run --script scripts/model-conversion/convert_parakeet_coreml.py \
  --workspace /path/to/parakeet-conversion-workspace \
  --output /path/to/parakeet-coreml
```

The result is a standalone runtime root. The schema-v3 package contains Core ML
encoder and decoder models, the decoder embedding table, and a 13-tensor MLX
fallback. It doesn't duplicate the full MLX encoder. Use it through `speech
transcribe --backend parakeet --provider coreml --coreml-encoder PATH`. The
runtime divides longer files into overlapping 15-second windows. It encodes
each window separately, applies the mel filterbank with Accelerate, and decodes
as many as 16 windows in parallel.

To measure the resident Release path, run the following command:

```bash
.build/release/mere.run model benchmark parakeet-coreml ./sample.wav \
  --artifact /path/to/parakeet-coreml \
  --warmups 2 \
  --repetitions 5 \
  --json
```

Generating the artifact doesn't qualify Neural Engine placement, speed,
boundary merging, or transcript parity. Profile and compare the compiled result
on each supported hardware target before publication.

## NVIDIA Nemotron 3 Nano Omni BF16 native layout

`mere.run model optimize` can stream NVIDIA's pinned 17-shard Nemotron Omni
BF16 checkpoint into a standalone native layout. The output keeps every tensor
payload exactly once: non-expert tensors retain their upstream keys across 17
slim shards, while 5,888 routed-expert tensors become 46 stacked MLX expert
tensors. No tensor value is quantized, rounded, or materialized during the
conversion.

```bash
mere.run model optimize /path/to/upstream-checkpoint \
  --output /path/to/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16-MLX-Native
```

Before publication, run the byte-for-byte validator. It compares every one of
the 7,349 source tensor payloads with the native partition and writes a
machine-readable verification receipt:

```bash
python3 scripts/model-conversion/validate_nemotron_omni_native.py \
  --source /path/to/upstream-checkpoint \
  --native /path/to/native-checkpoint \
  --output /path/to/native-checkpoint/NEMOTRON_OMNI_NATIVE_VALIDATION.json
```

The publishable artifact must also include the NVIDIA Open Model Agreement,
attribution notice, source revision, model card, and upstream safety documents.
The conversion is release tooling; inference downloads only the standalone
native checkpoint and never rebuilds an expert cache locally.

## Qwen3.8-Flash-Next MLX quantization

`convert_qwen38_flash_next_mlx.py` downloads Qwen's exact immutable BF16
revision once and streams its 131 safetensors shards into independent,
resumable MLX artifacts. It never instantiates the 180B-parameter model.

- `Qwen3.8-Flash-Next-MLX-4bit` applies affine Q4/group-64 to eligible
  language, MTP, and vision matrices. The 160-wide n-gram table necessarily
  uses Q4/group-32. Routers, norms, biases, convolutions, and incompatible
  shapes remain dense.
- `Qwen3.8-Flash-Next-MLX-Mixed-2bit` applies Q2/group-128 only to the 48 base
  routed-expert banks, keeps the n-gram table and eligible core/MTP matrices at
  Q4, and retains token/output embeddings, QSA indexers, routers, vision, and
  MTP fusion heads in BF16. This is the 128 GB Mac profile.
- `Qwen3.8-Flash-Next-MLX-Activation-3bit` creates fresh Q3/group-64 expert
  codes from BF16. It refits each accepted scale and bias against a frozen
  image and text activation profile. Eligible non-expert matrices remain Q4,
  and the n-gram table remains Q4/group-32.

All outputs retain the Qwen Community License 1.0 and include a complete source
and output hash receipt. Publication remains separate from conversion. The
activation-weighted Q3 release also includes a native qualification receipt
that records bounded text, image, Q4-control, memory, and swap results.

```bash
uv run --script scripts/model-conversion/convert_qwen38_flash_next_mlx.py \
  --workspace /workspace/qwen38-flash-next
```

To build the activation-weighted profile, provide the frozen activation file
whose SHA-256 is pinned by the converter:

```bash
uv run --script scripts/model-conversion/convert_qwen38_flash_next_mlx.py \
  --workspace /workspace/qwen38-flash-next \
  --profiles q3-activation \
  --activation-profile /workspace/q4-expert-input-second-moments.safetensors
```

To run the upstream MLX-VLM diagnostic, provide the model, fixture directory,
case manifests, and an output path:

```bash
uv run --script scripts/model-conversion/qualify_qwen38_flash_next_mlx.py \
  --model /workspace/qwen38-flash-next/Qwen3.8-Flash-Next-MLX-Activation-3bit \
  --external-ple-view /workspace/qwen38-flash-next/q3-external-ple \
  --fixtures /workspace/qualification-fixtures \
  --manifests /workspace/calibration/cases.json \
  --output /workspace/q3-mlx-vlm-qualification.json
```

Interrupted runs resume at the source-shard boundary. Do not publish an output
while its `.incomplete` marker exists.

## LiquidAI LFM2.5 QAD Q4_0 to MLX

`convert_lfm25_qad_mlx.py` converts LiquidAI's immutable QAD Q4_0 GGUF
checkpoints into native MLX safetensors for the existing Swift LFM2 runtime.
It repacks every Q4_0 nibble and FP16 block scale exactly into MLX affine
4-bit/group-32 tensors (`bias = -8 * scale`) without applying a second
quantizer. The GGUF's single Q6_K tied token embedding is decoded and
requantized to MLX affine 6-bit/group-64 so tied output projection remains on a
quantized kernel. The resulting maximum and mean elementwise errors are recorded
in `MERERUN_CONVERSION.json`.

Pass `--q4-layout affine64` to build the performance candidate that decodes the
QAD values once and requantizes them to MLX affine 4-bit/group-64. This is not a
bit-exact repack: its measured maximum and weighted-mean error are written to
the receipt, and it must pass output-quality and real-runtime speed gates before
publication. The default remains the exact group-32 representation.

The source directory must contain the pinned QAD GGUF plus the original model's
`LICENSE`, `README.md`, config, tokenizer, generation config, and chat template.
Every input is size- and SHA-256-verified before conversion. Outputs are written
transactionally and include a managed-model manifest, model card, and complete
artifact receipt.

```bash
uv run --script scripts/model-conversion/convert_lfm25_qad_mlx.py \
  --profile 1.2b \
  --source /path/to/lfm25-qad-1.2b \
  --output /path/to/LFM2.5-1.2B-Instruct-QAD-MLX-4bit

uv run --script scripts/model-conversion/convert_lfm25_qad_mlx.py \
  --profile 2.6b \
  --source /path/to/lfm25-qad-2.6b \
  --output /path/to/LFM2.5-2.6B-QAD-MLX-4bit
```

The publishable artifacts belong at
`Sawfwair/LFM2.5-1.2B-Instruct-QAD-MLX-4bit` and
`Sawfwair/LFM2.5-2.6B-QAD-MLX-4bit`. Pin the resulting immutable Hub commits in
the managed catalog only after native runtime and output-quality validation.

## NVIDIA Nemotron 3.5 Lightning and DSpark MLX

The Nemotron converters accept only NVIDIA's exact ModelOpt releases at the
pinned revisions embedded in each script. They verify every source file's byte
count and SHA-256 before conversion and write transactionally so a partial
artifact cannot be mistaken for a completed one.

The target converter repacks ModelOpt's uint8-paired E2M1 values into MLX's
uint32 NVFP4 container without changing a nibble, retains the E4M3 block and
FP32 global scales, and materializes 46 released FP8 Mamba projections as BF16.
It stacks the 128 routed experts per projection and omits the bundled MTP branch
in favor of the separately managed DSpark checkpoint. The companion converter
performs the same bit-preserving NVFP4 repack over its 20 quantized matrices.

```bash
uv run --script scripts/model-conversion/convert_nemotron35_lightning_mlx.py \
  --source /path/to/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --output /path/to/text-chat-nemotron-35-lightning

uv run --script scripts/model-conversion/convert_nemotron35_dspark_mlx.py \
  --source /path/to/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark \
  --output /path/to/text-chat-nemotron-35-lightning-dspark
```

Both outputs include `MERERUN_CONVERSION.json`, the original model card, and
the original OpenMDW-1.1 license. Python and ModelOpt formats are offline
conversion concerns only; inference is native Swift/MLX.

The managed target is published at
`Sawfwair/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-MLX`, pinned to commit
`6699e5fd3f0c5b392bb3f8bac2443276bb41958a`. Its DSpark companion is
`Sawfwair/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-MLX`, pinned to
commit `d30f0914d6bbb6da36302bd9228f92824901e675`.

## Muse Glimmer 30B MLX 4-bit

`convert_muse_glimmer_mlx.py` accepts only Meta's exact two-shard BF16 release
at revision `f84ecc3a0ea984a4c04542a84269e3d065350a6e`. It verifies the byte count
and SHA-256 of both weight shards, the index, tokenizer, template, processor,
license, usage policy, and model card before converting to native MLX affine
Q4/group-64. The default `selective` scope quantizes the 416 text-layer
projections, `lm_head`, vision adapter, and vision projection (420 matrices),
while retaining the token embedding and complete vision tower in BF16. This is
the quality-preserving release candidate. `--quantization-scope compact`
quantizes all 721 eligible matrices for a lower-memory experimental artifact.
Norms, biases, and the learned vision position table remain dense BF16 in both
variants. The transactionally written output includes a sharded index,
managed-model manifest, and a hashed `MERERUN_CONVERSION.json` receipt naming
the selected scope.

The managed selective artifact produced by this converter is published at
`Sawfwair/Muse-Glimmer-30B-MLX-4bit`, pinned to immutable revision
`6532e898dc5c1a55b51b1b108cd36728b79be751`. Its five remote shard sizes and
SHA-256 values match the bundled conversion receipt.

The source snapshot is about 59.6 GB. Conversion requires room for the source,
the new artifact, and temporary shards; inspect available storage before
starting. Review the bundled `LICENSE` and `USAGE_POLICY.md` first.

```bash
uv run --script scripts/model-conversion/convert_muse_glimmer_mlx.py \
  --source /path/to/meta-models--Muse-Glimmer-30B \
  --output /path/to/vision-chat-muse-glimmer-30b-q4
```

Build the compact comparison artifact explicitly:

```bash
uv run --script scripts/model-conversion/convert_muse_glimmer_mlx.py \
  --source /path/to/meta-models--Muse-Glimmer-30B \
  --output /path/to/vision-chat-muse-glimmer-30b-q4-compact \
  --quantization-scope compact
```

`convert_muse_glimmer_assistant_mlx.py` independently verifies the exact
5.11 GB DFlash assistant at revision
`2c86316d689027b91123638739743fef1d425233` and emits an approximately 1.44 GB
affine Q4/group-64 artifact with its own receipt:

```bash
uv run --script scripts/model-conversion/convert_muse_glimmer_assistant_mlx.py \
  --source /path/to/meta-models--Muse-Glimmer-30B-assistant \
  --output /path/to/vision-chat-muse-glimmer-30b-assistant-q4
```

The native runtime prefers the official BF16 assistant: the Q4 assistant was
slower at equal acceptance on the measured M4 Max MLX workload. This converter
exists for reproducible size/performance experiments, not as the default.

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
`61dc387ef1a7166425cdacd63c2340598dcc364f`. Alongside the transformer it
carries the pinned Qwen3-VL conditioner, video/audio VAEs, tokenizer,
`config.json`, `LICENSE`, `NOTICE`, `MODIFICATIONS.md`, source manifest,
conversion receipt, and hashes.
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

The compact package and staged residency design were informed by Phosphene's
[H3 engine notes](https://github.com/mrbizarro/Phosphene/blob/ee278e03acccdcce56cf2166608b25dfafaaa496/docs/H3_ENGINE.md)
and the pinned
[staged MLX runner](https://github.com/mrbizarro/minimax-h3-mlx/blob/c45a8d4da83eb943f174630751ed3261f86aa0ea/scripts/generate_staged.py).
Those projects are architecture references only: their weights, caches, and
conversion outputs are not artifact inputs.

The transformer core can remain BF16 or be quantized directly from official
BF16 to MLX affine Q8/group-64 (the legacy Q4 mode remains reproducible) while
precision-sensitive projections remain dense. Before conversion, all 52 fused
QKV matrices are deinterleaved from MiniMax's raw
per-head checkpoint rows into the official reference model's
`[all-q; all-k; all-v]` layout expected by the native runtime. A deterministic
QKV permutation fixture fails the conversion if that contract changes. The
Qwen3-VL conditioner is quantized directly from official BF16 to affine
Q8/group-64. Exact AdaLN tables are evaluated from the original released
projections for 5, 9, 12, 16, 21, and 31 points at video/audio shifts 12/3 and
the LightX2V 5-point shifts 6/3 schedule. The source-bound pack index records
each geometry, byte count, and SHA-256 before the schedule-only projections,
timestep MLP, and reconstructed RoPE tensors are omitted. The official
video VAE is cast to FP16 and the official audio VAE has weight normalization
folded exactly as required by the native runtime.

Production cache packs are evaluated by `mere.run model optimize` on MLX Metal.
CUDA and Metal can reduce the same BF16 matrix product in a different order, so
a CUDA-evaluated modulation table is structurally valid but not bit-identical
to live Apple Silicon inference. CUDA release builds therefore fail closed
unless both the Metal pack and its source-closure/real-parity receipt are
supplied. The independent validator requires the Metal receipt and exact 9- and
21-point real-generation hashes.

```bash
HF_HOME=/workspace/hf-cache HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
  python3 scripts/model-conversion/convert_minimax_h3_official_mlx.py \
    --cache-dir /workspace/hf-cache \
    --conversion-location "CA-MTL-3, Canada" \
    --transformer-precision bf16 \
    --metal-cache-pack /workspace/metal-cache-pack \
    --metal-cache-receipt /workspace/metal-cache-pack/adaln_cache.receipt.json \
    --output /workspace/minimax-h3-sawfwair
```

When only the transformer is rebuilt, compose it with the already validated
official-source conditioner and VAEs before publication:

```bash
python3 scripts/model-conversion/compose_minimax_h3_official_artifact.py \
  --base /workspace/previous-complete-artifact \
  --overlay /workspace/minimax-h3-transformer-v5 \
  --output /workspace/minimax-h3-release \
  --base-repository Sawfwair/MiniMax-H3-FL2VA-MLX-BF16 \
  --base-revision IMMUTABLE_BASE_REVISION \
  --base-transformer-precision bf16 \
  --transformer-precision bf16
```

The composer rehashes the previous complete bundle, hard-links or copies only
the three unchanged official-source components, overlays the reproduced core
and Metal cache files, and emits a complete source-bound receipt. It never
accepts Phosphene or another third-party model artifact.

The output carries `SOURCE_MANIFEST.json`, conversion metadata, the exact
upstream license and notices, and SHA-256 receipts for the publishable bundle.
Run the converter with `--plan` before provisioning storage, or with
`--self-test-only` to validate MLX's active backend before downloading weights.
Before publication, run the fail-closed structural and hash gate:

```bash
python3 scripts/model-conversion/validate_minimax_h3_official_artifact.py \
  /workspace/minimax-h3-sawfwair \
  --conversion-location "CA-MTL-3, Canada" \
  --transformer-precision bf16
```

The validator rehashes every distributed file and checks the complete official
source manifest, license bytes, Q4/Q8 and QKV fixture receipts, component tensor
counts, dtypes, retained 50-layer conditioner, Metal evaluation provenance,
every exact cache-pack entry, and forbidden omitted branches.

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
