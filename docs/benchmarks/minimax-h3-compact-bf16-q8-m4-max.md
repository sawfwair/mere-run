# MiniMax-H3 compact BF16 and Q8 on M4 Max

This receipt records the official-source compact-artifact rollout and real
native Swift/MLX inference on an Apple M4 Max with 128 GB unified memory. The
test media uses 416 x 256 pixels, 22 frames at 24 frames per second, stereo AAC
at 32 kHz, and matched
prompts/seeds so several lanes could be reviewed without projecting a large
production render.

## Artifact identity

- **Official source:** `MiniMaxAI/MiniMax-H3@ec19cc6daf5d8add9417c18e86b6b58cc6c55027`
- **Official source bytes:** `144,035,116,604` across 48 files.
- **Cache-producing source tensors:** 106 tensors and `26,142,079,488` bytes.
- **Cache source closure SHA-256:**
  `e2ccc0cab72b9183a0347e3999f4559cdc315b7b363a5fe9196890dd315f5a40`
- **Compact BF16:** `Sawfwair/MiniMax-H3-FL2VA-MLX-BF16@6f2c1edb4d31d9110d4a51457ba1d6401a05dfd0`.
- **BF16 managed bytes:** `76,861,026,073`.
- **Affine Q8/group-64:** `Sawfwair/MiniMax-H3-FL2VA-MLX-8bit@57a926c2422e09c8563cd2e0c43b2e94ef791de4`.
- **Q8 managed bytes:** `58,075,175,639`.
- **Rollback source:** `Sawfwair/MiniMax-H3-FL2VA-MLX-BF16@c768b13a964f646a8000e641608189289e4514af`.

These identities remain the historical roots used for the measurements below.
The managed BF16 and Q8 pins were later refreshed to `4ce4b1d870f7b1b0c75672fd4f2867c1f5df7b5f`
and `86500cb6ebec22c006597e41840b26ef1099fdd7` to add the exact LightX2V
nine-point shifts-6/3 AdaLN table; the benchmark results weren't reattributed
to those newer package revisions.

Both denoising cores were reproduced from the pinned official checkpoint on an
ephemeral RTX A6000 host in Sweden with MLX/MLX CUDA 0.29.3. The BF16 core is
`40,138,395,701` bytes with SHA-256
`86e6f89bdf7c5d6c49a8b7dfeb4b974599c7a83d740e7e1b6333009c627cd20d`.
The Q8 core is `21,352,545,067` bytes with SHA-256
`b5fc7dbc0efb55041be3c84447acf6eccea7faf3df572b56378497d61a692177`.
The independent validator confirmed 428 unquantized BF16 transformer tensors
and 844 Q8 tensors, including 208 affine group-64 linears, with all 51 AdaLN
projections, the timestep MLP, and reconstructed RoPE tensors absent.
After publication, fresh managed pulls were independently revalidated from the
same roots used for inference: BF16 closed at `76,861,026,073` bytes across 21
files and Q8 at `58,075,175,639` bytes across 21 files.

The cache pack was not evaluated on CUDA. `mere.run model optimize` generated
the seven exact schedule tables on MLX Metal, and the release converter imported
the source-bound pack. The pack index SHA-256 is
`aa129cb8de43d0218e600bfad9a8a42acd431a108adb9d48135f50ee1401089c`.
CUDA and Metal produced structurally valid but non-bit-identical BF16 AdaLN
tables, so converter version 5 and the validator fail closed unless the release
contains the Metal receipt and exact 9-/21-point media parity.

## Exact BF16 and adapter parity

The old full BF16 root and compact BF16 candidate used the same release binary,
prompt, seed, geometry, schedule, resident-BF16 policy, and compiled execution.

| Workload | Full and compact MP4 SHA-256 | Video and audio result |
| --- | --- | --- |
| Base, 9 points | `29c76863cae58644d3747e7fd362d46877fdc7c31c5dc60d8be98d4651843f21` | SSIM 1.0, PSNR infinite, audio correlation 1.0, audio relative L2 0 |
| Base, 21 points | `5449b7e66b803f71d4e78d558f1710b6558b684ff722f0155992df6d94f05ad1` | SSIM 1.0, PSNR infinite, audio correlation 1.0, audio relative L2 0 |
| Larry Turbo, 5 points | `e37e5f927a729ec920b0b3bbc5f605f59cfbd5ca80df841c10ed4c3db907bd19` | byte-identical MP4 |
| LightX2V, 5 points | `4d6662833f9bb42ff508725352ecc5889a92fb4b46d29383c16fffc4f9ba0873` | byte-identical MP4 |

Larry's 51 AdaLN LoRAs were applied to the cached modulation activations in
memory. LightX2V used activation-space dense/Q8 LoRA wrappers, including the
independent global-slab Q/K/V path; no 26 GB AdaLN base restoration, fused
adapter checkpoint, custom quantized kernel, or persisted augmented cache was
used.

The published BF16 managed root was then exercised independently of the parity
staging roots. Its base, Larry, and LightX2V MP4 SHA-256 values were
`748a77ce065d8efeb85a186f0fd226e824f277af8c301e8b527119fbfe147b3b`,
`1dbfb7e01cad686dc798578a510d0e9e014920277adbbf9ad85846fa078634a2`,
and `ad45678a374de7c27e71833be503cbdadf28413d7f1672bc2f550ddedbc781a7`.
All three outputs contained 416 x 256 H.264 video and stereo 32 kHz AAC audio,
and all three processes reported zero swaps.

## Q8 quality compared with BF16 and rejected Q4

Three matched prompt/seed lanes covered character identity/dialogue, rally-car
camera motion, and lighthouse fine texture/weather. Contact-sheet review found
no Q8 collapse, lattice, identity loss, broken motion, or additional audio failure.
Q4 changed the character into a helmeted figure and materially changed motion,
background, and lighthouse structure. This was an operator review rather than
an independent blinded panel, so the latter remains a release-note limitation.

| Lane | Candidate | SSIM | PSNR | VMAF | Audio corr. | Audio rel. L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Lighthouse | Q8 | 0.921044 | 29.082862 | 36.460866 | 0.958130 | 0.333067 |
| Lighthouse | Q4 | 0.892674 | 26.037928 | 19.447434 | 0.756427 | 0.687420 |
| Character | Q8 | 0.904596 | 29.178586 | 42.299904 | 0.998573 | 0.062542 |
| Character | Q4 | 0.665353 | 18.742957 | 0.392989 | 0.271018 | 1.232457 |
| Motion | Q8 | 0.908233 | 30.141434 | 51.893139 | 0.984433 | 0.176275 |
| Motion | Q4 | 0.699028 | 18.835835 | 0.022313 | 0.681862 | 0.792367 |

The Q8 character, motion, and lighthouse MP4 hashes are respectively
`ab086744b7ee5c04d80eca7c7bf1c9c432deaef9254c3f8fa4e0e447bc9359ec`,
`0f21ff089d62bdf49c14a5e33e6a4cf2bd27855d1ca567b69c26bd93d1b3edd4`,
and `c5f15068c5bd2e905c7669095a087f68794377ef620c3a7488da8d5846ba253d`.
The Q4 controls are
`90a1eeecb20fa38e0695af14f3ac490b07e96e0fe00caccebfc07de795178111`,
`e5a1a05996d7a2bfc8274ac72cbbc23e0343b24fd4d7971424e4ca5aefcb6320`,
and `f92dcf5afb9a3496db4886b8ccf8c8b3073959b563e027db88951c0df769ad52`.

The published Q8 managed root also completed base and LightX2V inference. The
base MP4 SHA-256 was
`ad1a46b243e427d9b5ea25caaf9244c8461af2b09d68a7646477a1308b489c1f`;
it took 35.809 seconds end to end, including 31.884 seconds of denoising, with
a 27.63 GB maximum RSS and zero swaps. The activation-space Q8 LightX2V MP4
SHA-256 was
`06aab6eabe4989a15e90b861a7430e0168cda8d5ef69b3360944399bbf4f7168`;
it took 20.859 seconds end to end, including 16.320 seconds of denoising, with
a 29.01 GB maximum RSS and zero swaps. Both files had the expected H.264/AAC
streams, and contact-sheet inspection found coherent lighthouse structure and
motion without lattice or collapse.

## Performance and memory

The most comparable 9-point BF16 runs measured 38.698 seconds for the full
root and 38.542 seconds for compact BF16 denoising, a 0.4% improvement. Larry
measured 28.943 versus 19.521 seconds, and LightX2V measured 18.929 versus
18.070 seconds. Those adapter pairs show no observed 3% runtime LoRA
steady-state penalty. They are not generalized speed claims.

The external ExFAT candidate made cold preparation and end-to-end numbers
storage-bound: compact BF16 preparation took about 97 seconds from the external
drive versus 11.7 seconds for the internal full root. A same-storage cold-read
comparison after migration copied only the `40,138,395,701`-byte compact core to
internal APFS and left the remaining components linked to the published root.
Transformer preparation took 6.172 seconds versus 11.715 seconds for the full
root, a 47.3% reduction, and generated the same
`748a77ce065d8efeb85a186f0fd226e824f277af8c301e8b527119fbfe147b3b`
MP4. The OS denied the unprivileged disk-buffer purge, so this is an honest
same-storage warm-read result rather than a cold-read claim. The temporary
40.14 GB internal core was removed immediately after measurement.
The 21-point compact run was also host-contention affected: text encoding took
66.8 seconds and decode 11.8 seconds versus 4.1 and 2.9 seconds in the control,
so its 105.116-versus-98.645-second denoise pair is not used as a performance
gate. Background app/relay processes remained active during these correctness
runs.

Q8 reported about 20.17 GiB active and 25.61 GiB peak Metal memory. The process
maximum RSS was 27.63 GB with a 27.8 GB peak footprint and zero process-level
swap events. BF16 reported about 38 GiB active and 40.9 GiB peak Metal memory.
Q8 transformer setup was 0.149 seconds, but lazy reads made the first denoise
step 49.250 seconds; later steps were 3.9-5.23 seconds. Q8 is therefore promoted
for disk, RAM, and quality rather than speed.

## Managed migration and storage

Before migration, the full internal BF16 root reported `100,323,441,381`
logical bytes and occupied `100,323,647,488` allocated bytes.
The SALVATION Hub cache already held both candidate cores and shared components
once, using about 98.3 GB physically; it had about 27.7 GB free plus a preserved
21.9 GB build-cache parking reserve during artifact construction. Immediately
before the transactional pull, internal available space was `18,709,772` KiB;
afterward it was `83,435,384` KiB, a measured increase of `64,725,612` KiB
(`66,279,026,688` bytes). The reclaimed amount is lower than the old root's
allocated size because APFS had clone-shared blocks with other local payloads.
SALVATION changed by only 128 KiB during the pull because every payload blob was
already present. The earlier internal directory and all staging and previous
siblings are gone. The BF16 and Q8 managed roots occupy 4 KiB each and resolve all 21
payload files through absolute links to SALVATION. Shared conditioner, VAE,
tokenizer, license, and cache-table links resolve to identical content-addressed
blob paths for both models.
`model pull --cache-dir /Volumes/SALVATION/MereRun/hub` prepared each
revision completely, validated a sibling staged root, atomically exchanged the
managed install, revalidated aliases/source identity, and only then removed the
previous root. The internal managed roots resolve through absolute symlinks to
content-addressed SALVATION blobs; disconnecting the volume makes them
unavailable.

## Commands and remaining boundaries

The following commands reproduce the preparation and validation steps:

```bash
mere.run model optimize /tmp/h3-adaln-rebuild --json
python3 scripts/model-conversion/hash_minimax_h3_adaln_source.py OFFICIAL_ROOT
python3 scripts/model-conversion/validate_minimax_h3_official_artifact.py \
  ARTIFACT_ROOT --conversion-location "SE, Sweden" --transformer-precision bf16
mere.run model pull video-minimax-h3-fl2va-bf16-mlx \
  --cache-dir /Volumes/SALVATION/MereRun/hub --accept-model-license
mere.run model pull video-minimax-h3-fl2va-8bit-mlx \
  --cache-dir /Volumes/SALVATION/MereRun/hub --accept-model-license
env -u MERERUN_RUN_E2E nice -n 10 ./scripts/check.sh
```

The real Q8 runs were on a 128 GB host, not a 64 GB machine. The catalog keeps
the 96 GB minimum and makes no 48/64 GB claim. Automatic transcription was
initially blocked by the pre-migration 16 GB disk-reserve gate. After migration,
the installed Parakeet backend transcribed the matched BF16, Q8, and Q4
character clips identically as `We made it home.` with exit status 0 in all
three lanes. `/Applications/MereRun.app`
remained v0.41.0 build 781; its helper and app executable SHA-256 values stayed
`1d046530b2a406f36b411abf1b2f9d9ad5b6724d3c9b4f9442946f1e64cb8efe`
and `f541290449092cf9eaf5201ae0eec0fe2c0f17fd9b92fc2b1d2360d673573272`.
