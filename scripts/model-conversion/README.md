# Native model conversion

These scripts are audited release tools. They are never invoked by a
`mere.run` inference command and never become a Python sidecar.

The VDA-S, TripoSR, and InstantMesh converters have one frozen serialization
environment: CPython 3.11.15 with the exact packages in
`requirements-vfx.txt`. They fail closed on any other tool version and verify
the byte count and SHA-256 of every emitted weights, configuration, provenance,
and license file before publishing the staged directory. Their `--license-file`
argument is mandatory and accepts only the exact pinned upstream license bytes.

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
