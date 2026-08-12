# MiniMax-H3 Ref2VA MLX 8-bit validation

This receipt records the corrected managed Ref2VA artifact and one real native
Swift/MLX generation on an Apple M4 Max MacBook Pro with 128 GB unified memory.
It is correctness and usability evidence, not a claim of speed parity with
FL2VA or LTX.

## Artifact identity

- managed ID: `video-minimax-h3-ref2va-mlx`
- repository: `Sawfwair/MiniMax-H3-Ref2VA-MLX-8bit`
- immutable revision: `61dc387ef1a7166425cdacd63c2340598dcc364f`
- managed bytes: `70,941,103,245`
- source: `Comfy-Org/MiniMax-H3@fd70b39279d1ae6eb214c903f53e1bec3af19a77`
- source file bytes: `34,038,894,550`
- source SHA-256: `9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9`
- converted transformer bytes: `36,024,412,656`
- converted transformer SHA-256:
  `234f22f69f8d40d6ed81cceed8259fa287f3c9417d40fba5274e3a7aa84e18a2`

The transformer and Qwen3-VL conditioner are MLX affine INT8/group-64. The
video VAE is FP16 and the audio VAE is FP32. Lower Ref2VA precision did not
meet the visual quality bar, so 8-bit is the published floor.

The 14-file managed pull also includes `adaln_cache.safetensors`, exactly
873,820,740 bytes with SHA-256
`2cbe9e3324ef2cc5108a3ba7f1219d84079ff00a017f604fd86300005cc64fcd`.
Its source identity is the immutable converted-transformer SHA-256. The cache
was built one three-modality timestep batch per released schedule point, then
reloaded and compared with both a fresh cache and direct live AdaLN evaluation.
At schedule step 10, maximum video and audio output error were both zero.

### Managed revision-upgrade receipt

The artifact was first installed at pre-cache revision
`abb9114fe9d6e3cccc6376eee1abaf09d3f2a9fe`, with the corrected cache added to
the model root locally. After the catalog pin moved to the revision above,
ordinary `model pull` now treats the old manifest as stale and asks the Hub
tree for the target revision's Git/LFS object identities. It preserves the
installed root while preparing the new immutable snapshot and adopts a local
payload only after its byte count and Git blob SHA-1 or LFS SHA-256 match the
target object.

On that real stale install, structured preflight reported 4,362 bytes rather
than the package's 70.94 GB logical size. Those bytes were the two changed
managed provenance files; every large target object, including the
873,820,740-byte cache, was already available by exact content identity. A
normal pull, without `--force` or a preparation command, produced a 14-file
snapshot receipt whose requested and resolved revision are both
`61dc387ef1a7166425cdacd63c2340598dcc364f`. Post-pull model validation checks
the cache byte count and safetensors format, schema, and transformer-bound
source identity in addition to the conversion receipt.

## Conversion correction

The source checkpoint stores per-tensor ConvRot metadata. Two different source
rotation groups are present: group 256 for 200 transformer matrices and group
64 for 50 AdaLN matrices. The faulty conversion treated the MLX output group
size as though it were also the source rotation group, producing noise.

The corrected converter reads and validates every tensor's embedded source
group, reverses that regular-Hadamard basis, and only then requantizes the
restored matrix to MLX affine INT8/group-64. Unit fixtures cover both source
groups and mixed-group conversion. The final artifact records the toolchain,
source identity, group counts, quantization, output size, and hash in
`transformer.conversion.json`.

Numerical checks on the corrected route measured approximately `1.729e-6`
relative L2 error before MLX requantization and `0.0073349` for a packed linear
operation after INT8 requantization.

## Native generation receipt

The full validation used real reference conditioning and maximum acceleration.
It produced:

- 512x320 H.264 video;
- 124 frames at 24 fps (`5.167` seconds);
- AAC stereo audio at 32 kHz;
- wall time: `1,724.17` seconds;
- MP4 SHA-256:
  `08ce4cfb9fe305ba67297d94f6a54f3ce49c12bb8e242c38561cb6cb6237c9b0`.

The output was visually coherent, retained usable motion and subject structure,
and generated intelligible synchronized dialogue transcribed as: “You kept up
the recording? Yeah. Every second.” A preceding 256x160, 22-frame smoke was
also coherent. This establishes that the corrected artifact executes the real
Ref2VA path without the all-noise failure.

The 28-minute wall time confirms that this path remains substantially slower
than desirable. The result should therefore enter the planned LTX/H3 bake-off
as a quality and conditioning candidate, with performance measured separately.
