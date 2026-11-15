# SOV automatic expert reduction — 2026-11-15

Q8 Metal inference now selects the qualified distributed expert reduction
automatically on an **Apple M5 Max** with GPU architecture **`applegpu_g17s`**.
Users do not need `--experimental-expert-reduction` on that configuration.
Both device name and architecture must match. Other devices use the original
SIMD reduction; CPU execution does not select Metal, and unsupported projection
shapes keep their existing fallback. Expert precision stays as selected.

The Metal algorithm is unchanged from the
[160-output qualification](./sov-transcribe-distributed-reduction-2026-11-14.md).
Only its header comment changes. The receipt verifies that the shader text,
excluding line comments, matches the archived qualified source exactly.
This pass changes dispatch policy and reporting, not quantization or arithmetic.

## Selection and controls

`Gemma4MetalExperts` resolves the hardware default when constructed. A nil
reduction override restores automatic selection; explicit true and false still
force either policy for research. Fusion-only setup preserves the automatic
default. Retained compilation identities continue to distinguish each resolved
kernel policy.

Normal Q8 Metal use, including the speech CLI's `--sov-decoder-experts q8-metal`
selection, needs no additional reduction option. BF16 and plain MLX Q8 retain
their existing decoder selection. The benchmark force flag remains compatible.
Paired benchmark controls explicitly disable distributed reduction, while the
candidate uses the qualified default unless forced. `+expert-reduction` labels
reflect the selected policy rather than whether the force flag was supplied.

## Validation

The normal focused debug GPU/runtime/CLI/documentation selection passes:
**43 tests passed, one optional residency test skipped**. New tests cover exact
device/architecture matching, unknown and future devices, CPU fallback, automatic
projection bits against an explicit reference, model dispatch overrides,
detach/reattach, and preserving automatic selection with fusion enabled.
Existing width, tail, dense-prefill, parameter-update, and decoder checks pass.

The release CLI builds successfully. Five fresh native smoke processes verify
**14 measured outputs** against frozen token IDs and DFlash acceptance lengths:

- Short, long, and wrapped recordings run without the force flag: two outputs
  each, all labeled with the automatically selected reduction.
- A short paired comparison against MLX runs without the force flag: four
  outputs, with reduction selected only for the Metal candidate.
- A short paired comparison against the original SIMD layout uses the retained
  force flag: four outputs, with the control pinned to the original reduction.

These are dispatch and parity checks, not a new performance qualification.
The [previous paired results](./sov-transcribe-distributed-reduction-2026-11-14.md)
remain the performance evidence. Tests of unknown devices exercise the selector;
this pass does not claim execution qualification on another GPU.

Strict lint without its cache, Python syntax, and `git diff --check` pass.
The full repository gate remains deferred at the user's request. An attempted
release-mode test build is blocked by an unrelated existing reference to
`MiniMaxH3FastVSAKernelMode.fullTileKV16` in
`Tests/H3RuntimeTests/H3ComputationTests.swift:191`. A direct release test-target
attempt also encounters a module compatibility error. Those files are unchanged;
the normal debug test workflow and the actual release CLI validate this change.

The [JSON receipt](./sov-transcribe-automatic-reduction-2026-11-15.json) contains
the hardware identity, exact source/binary hashes, shader comparison, smoke
commands, raw reports, frozen-reference checks, and validation gaps.
Artifacts and a verified hash manifest are archived on the external SSD under
`/Volumes/SALVATION/MereRunProfiles/sov-automatic-reduction-2026-11-15/`.
