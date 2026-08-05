# ACE-Step Validation

This is the evidence contract for the native Swift/MLX ACE-Step 1.5 runtime.
It separates source parity, local tests, installed-checkpoint execution,
performance, and listening review.

## Immutable sources

The implementation was compared with ACE-Step upstream commit
`6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`. Managed downloads use immutable
Hugging Face revisions:

| Component | Repository | Revision |
| --- | --- | --- |
| Shared VAE/text/Turbo assets | `ACE-Step/Ace-Step1.5` | `19671f406d603126926c1b7e2adc169acbcade22` |
| 1.7B planner | `ACE-Step/Ace-Step1.5` | `19671f406d603126926c1b7e2adc169acbcade22` |
| XL-Turbo | `ACE-Step/acestep-v15-xl-turbo` | `d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee` |
| 4B planner | `ACE-Step/acestep-5Hz-lm-4B` | `0a3ec94b557aea7d508da38b31cfe7341f6ff737` |
| XL-SFT | `ACE-Step/acestep-v15-xl-sft` | `d06de46b4622f781cf07f4a013a67d591ca52819` |
| XL-Base | `ACE-Step/acestep-v15-xl-base` | `220c1166efbdd9583eafcb12eb160594bbfcb241` |

Generation recipes repeat the effective repository/revision set, adapter
SHA-256 and scale, final effective conditioning metadata, complete inference
configuration, candidate ranking, and input/output hashes. Recipe schema 4
adds planner temperature, top-k/top-p, and repetition penalty to schema 3's
resolved planner source, root, and immutable repository/revision provenance
and schema 2's post-planning BPM, duration, key/scale, vocal language, and time
signature. This protects old installs whose original
manifest predates source provenance, distinguishes requested metadata from the
values actually used, and records independently resolved DiT and planner
components.

## Parity and local gates

The deterministic test surface covers:

- Turbo and continuous timestep schedules, Euler/Heun updates, CFG, APG, ADG,
  guidance windows, and stabilization;
- typed task/checkpoint capability routing and exact Base-only restrictions;
- source conditioning, repaint masks/injection/splice, retake endpoints and
  spherical interpolation, and flow-edit integration windows;
- prompt/condition encoder, audio-tokenizer, DCW, first-velocity, final-latent,
  and VAE parity dump paths;
- Python-contract metadata precedence: explicit duration and language survive
  conflicting planner proposals, 1.7B is discovered before 4B, and separately
  managed planners resolve without checkpoint-store symlinks;
- stable seed fanout, candidate metrics/ranking, batch/session serialization,
  API decoding/security, WAV headers, LRC, recipes, and DAW bundle topology;
- PEFT LoRA and LyCORIS LoKr key mapping, numerical contributions, alpha/scale,
  decomposed Kronecker factors, and adapter stacking.

Pinned-upstream numerical probes isolate every core stage. On the installed
XL-Turbo stack, Swift versus native MLX measured mean absolute errors of
`9.74e-7` for DiT, `2.08e-7` for the condition encoder, `8.21e-6` for Qwen
hidden states, `6.05e-7` for FSQ detokenization, and `6.32e-7` for VAE decode.
The VAE also passes a real structured-audio round trip. This stage isolation
identified the prior output failure as orchestration rather than checkpoint or
kernel drift: Swift supplied a float chunk mask of `2.0`, while upstream stores
that assignment in a boolean tensor and therefore conditions the model with
`1.0`.

Run the normal repository contract:

```bash
./scripts/check.sh
```

Installed-checkpoint tests are opt-in because they require large local assets.
The integration tests describe the required `MERERUN_TEST_ACESTEP_*`
environment variables and skip truthfully when an asset is absent.

## Installed-model evidence

All files below were generated locally at 48 kHz stereo. Hashes make the
specific validation artifacts identifiable without committing model output.

| Path exercised | Configuration | Result |
| --- | --- | --- |
| XL-SFT | 50-step Heun, APG, final quality | `979775ce2503c7c393b3fc435a45515ffb4be80a3658e26d90b541cc57a888a4` |
| XL-Base extract | one-second source, one step, `Drums` | `9c620ce47d89b67824c2eebd88c20c0a81b1c340355a2ddd0937c61db0f78851` |
| XL-Base lego | one-second source, one step, `Drums`, 0.2–0.8 s mask | `322d3c757da5477ec3a828d2d198abcb80c9a3ff14a20a3c2786119a6dacb171` |
| XL-Base complete | one-second source, one step, `Drums,Bass` | `1c8861285bcbf0952035b11aa1676c38f516a543c700c15a896105c0f9fdea35` |
| XL-Turbo flow edit | one-second source, one step | `ae758bdf4b940277fc09145f9340bdd852b1fd48915921ad231b86488353cb27` |
| XL-Turbo + 4B recipe v2 | one-second LM-planned vocal, one step | `68b170220cf5849635f3f97f2b75d87d7a80045f2401edf9a707d677db2f2122` |
| Resident API WAV | warm one-second XL-Turbo request, one step | `9c6f3ef3752f15636aa2f4eb7b74b3dcee556e78a62158e4bc4ac344f6eaee32` |
| Float32 artifact/DAW workflow | recipe, LRC, candidates, bundle | `d44d8386c6b479e209d169c74ff2aa69bd2b13dd980f83a708b9f381c985e629` |
| LoRA train/save | one real backward and AdamW step, 256 layers | `b7c9c40c3b74aca7fb8f12fb4999c0be1175db82bf9e1504bb9c19794a29c8dd` |
| LoRA reload/generate | trained artifact applied to 256 layers | `1ba7876df682c626b1f08da80a739ed0685becdf0182e492bb1c533d07d41449` |
| LoKr train/save | one real backward and AdamW step, 256 layers | `5175420753533a736b43afa608dfc957e48c8d216ac0ec0d6ac824030e1c739e` |
| LoKr reload/generate | factored adapter applied to 256 layers | `79a45e0d340ad0232bb81f4c68ff5c5cc57fc9541f11343f5171af0a319e1961` |
| Provenance recipe | exact pins plus adapter hash and scale | `6c3370f61c3ed3bea3fc48e99369432d1578696be2989d3713abbc094a68a2bb` |

The XL-SFT 50-step run completed in 15.52 seconds with 20.72 GB maximum RSS,
25.34 GB peak footprint, and no swap. The XL-Turbo flow edit completed in
4.98 seconds with 21.03 GB maximum RSS, 26.94 GB peak footprint, and no swap.
The XL-Base extract, lego, and complete smokes loaded the immutable Base
checkpoint and produced PCM24 output in 3.27, 2.31, and 4.87 seconds. Extract
and lego peaked near 26.1 GB; complete peaked at 48.34 GB. All three completed
without swap. The temporary Base installation was then removed through
`mere.run model remove`, reclaiming 19,949,344,762 bytes while retaining shared
assets still referenced by installed ACE-Step variants.
The one-step LoRA train/save smoke took 6.52 seconds with 21.02 GB maximum RSS
and 25.84 GB peak footprint. LoKr took 2.97 seconds with 21.18 GB maximum RSS
and 26.15 GB peak footprint. Both adapter reload renders completed in roughly
2.3 seconds without swap. The real 4B-planned schema 2 recipe retained the
effective BPM and explicit one-second duration while discarding unsupported
planner language output.

The real resident API health probe reported XL-Turbo + 4B loaded and warm. A
one-second PCM24 WAV request returned in 0.68 seconds. A heterogeneous batch
returned independent candidate counts `[1, 2]`, deterministic seeds `[6201]`
and `[6301, 6302]`, and exactly one selected candidate per item in 1.16
seconds. The response preserved per-item BPM, duration, key, and time-signature
metadata. A real LM-planned API request returned effective BPM 108, G major,
the explicit one-second duration, and omitted unsupported planner language.
Wrong-model, wrong-content-type, and raw-WAV batch requests returned actionable
HTTP 400 JSON errors.

## Listening regression

`Tests/MereRunCLITests/Fixtures/ACEStep/listening-regression.json` freezes
prompts, lyrics, seeds, duration, quality, candidate count, and listening
criteria across transient electronic, acoustic, vocal, dense, and ambient
material.

```bash
./scripts/acestep-listening-regression.sh
```

The runner writes the WAVs and exact recipes, SHA manifest, ordered M3U
playlist, and a review CSV with structure, prompt adherence, audio quality,
vocal alignment, regression, and notes fields. The automatic gate catches
silence, clipping, DC, stationary broadband noise, missing time-varying
spectral structure, and prematurely dead endings. The CSV deliberately
retains a human listening decision for musical quality and prompt adherence.

The installed XL-Turbo + 4B run completed all five frozen cases as 48 kHz
stereo PCM24. Each selected candidate used all 60 semantic audio codes expected
for 12 seconds and passed `temporalSpectralVariation >= 0.85` plus
`tailEnergyRatio >= 0.05`:

| Case | Score | Spectral variation | Tail ratio | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| transient electronic | 86.7512 | 1.3137 | 0.0720 | `1a5c7c56e5373d69400f553256637199bb2b1e04877435abf8f2e2ebe6e555dc` |
| acoustic space | 90.2295 | 1.2200 | 0.5534 | `bb4a9a1e5a208bb6d594a83820ffd330c98e2d83aca1f932b7429572192dc3b7` |
| vocal alignment | 87.2707 | 0.9795 | 0.9391 | `781616e86a5947f46383f90744d69161dffb087d0f90e363e72b36fa62bb1b2c` |
| dense arrangement | 92.6983 | 1.3598 | 1.2525 | `91ca6ad435b7f5ee12a550aa0e19da3348bbfce0c805b110159ed0363680a704` |
| ambient continuity | 90.0086 | 1.1366 | 0.9754 | `356379ebeffdbeb7b50dd1e3f7062a00bfd742174d3a698cd4d74585a6a25391` |

Repeating the transient case with the same seed reproduced its planner
metadata, candidate scores/code counts, and WAV byte-for-byte at SHA-256
`1a5c7c56e5373d69400f553256637199bb2b1e04877435abf8f2e2ebe6e555dc`.

The review CSV remains deliberately unscored until a person listens. The
technical rank is evidence against the exact stationary-noise regression, but
it is not a substitute for a musical-quality or lyric-alignment judgment.

For a repeatable timing capture:

```bash
./scripts/acestep-performance-proof.sh
```

The installed one-second/one-step XL-Turbo proof completed in 3.14 seconds,
including process startup and checkpoint load, with 21.35 GB maximum RSS,
25.35 GB peak footprint, and zero swap. It produced PCM24 stereo at 48 kHz with
SHA-256
`a5d5a265488373d9b32beabd1ccdf5c7e99cc4d159b08a1a8aaa006ef12bdf1f`
and a matching schema 2 recipe.

Do not treat a source-level test, generated recipe, or unchecked listening
playlist as installed-model proof. Each evidence layer answers a different
question.
