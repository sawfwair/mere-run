# Bonsai 2 Small upstream parity with precise Metal frequencies — 2026-10-16

The native 2B, 4B, and 9B implementations now pass every frozen gate across the same 18 text/vision cases and 930 teacher-forced continuation positions used in the [October 15 comparison](./bonsai2-small-upstream-parity-2026-10-15.md). The correction uses an explicit precise Metal power intrinsic for checkpoint vision rotary frequencies. Checkpoints, upstream references, input corpus, and tolerances are unchanged.

## Cause and correction

A standalone Metal probe evaluated the frequency expression with fast and precise math functions under Metal 3.2 and 4.0. With safe arithmetic, fast `pow` reproduced the previous 2B native inverse-frequency bits exactly and differed from upstream in 11 of 16 entries. Precise `pow` reproduced upstream exactly for both frequency dimensions, 32 and 36. Changing the language version alone did not remove the discrepancy.

The native GPU path now calls `metal::precise::pow` through a small MLX custom kernel. Reciprocal and position multiplication remain separate MLX operations, preserving their rounding. This is a general mathematical operation, without stored checkpoint frequencies or case-specific outputs. CPU execution retains MLX power. The opt-in checkpoint arithmetic policy scopes the change to Prism vision towers; no vendor artifact or package dependency changed.

The independent rotary fixture now requires bitwise equality on CPU and GPU, replacing its previous GPU-only component tolerance. Full-checkpoint gates remain unchanged: exact prompt IDs, image grids, mRoPE positions, and greedy decisions; normalized-patch maximum absolute delta at most `1e-6`; image-embedding relative L2 at most `0.01`; maximum conditional probability delta at most `0.01`.

## Results

The following table shows each model's largest differences and gate result.

| Model | Image embedding max relative L2 | Max probability delta | Greedy decisions | All gates |
| --- | ---: | ---: | ---: | --- |
| 2B | 0.000000000 | 0.000012553 | 331/331 | Pass |
| 4B | 0.000000000 | 0.000004563 | 293/293 | Pass |
| 9B | 0.005305753 | 0.002422046 | 306/306 | Pass |

All eight 2B/4B image embeddings are now bitwise identical to the independent upstream outputs. This resolves their previous failed embedding gates and the 2B shapes probability failure. The 9B model retains small vision numerical differences within the unchanged gate; its maximum image-embedding relative L2 is about 0.53%. No bitwise 9B vision claim is made.

The comparison independently preprocesses and tokenizes native requests and exports full-vocabulary conditional logits under the upstream greedy continuation. Each model receives two text prompts and four image prompts, with up to 64 continuation positions per case, context 16,384, and prefill chunks of 512. The pinned source identities, corpus, tensor hashes, executable receipt, and individual gates are in the [machine-readable summary](./bonsai2-small-upstream-parity-2026-10-16-summary.json).

## Evidence and limits

New artifacts are under `/Volumes/T7/mere-run-qualification/bonsai2-small-2026-10-15/parity-precise-2026-10-16`. The previous failed comparisons remain intact under `parity`. Immutable reference exports are reused through links; all three native models loaded their original pinned checkpoints and acquired standard shared admission independently.

The full-model result belongs to the preserved test executable and recorded source hashes. Release smoke and the final local gate are recorded in the following section. This bounded comparison does not establish broad quality, sampled-output equivalence, maximum-context accuracy, or cross-device behavior. The original 16-case quality corpus and resident throughput were subsequently rerun for the corrected release; see the [fixed-runtime quality and throughput report](./bonsai2-small-fixed-2026-10-16.md).

## Release smoke and local validation

The release build passed. Its SHA-256 is `6468cd000e7268c9ba1678892a96beccbf13f296fceacf970a4af7f72322d23a`; the tested executable and its preserved T7 copy match. All three CLI chart requests return PINE 9, CEDAR 4, and difference 5 under greedy sampling, thinking disabled, context 16,384, and a 256-token budget. All three wrap the JSON in Markdown fences, as upstream does: semantic scoring passes 3/3 and strict JSON scoring passes 0/3. These fresh-process checks are separate from resident throughput.

The five focused GPU tests pass, including exact independent frequency fixtures for both head sizes. The complete repository gate (`./scripts/check.sh`) passed after unlocking: 5,472 XCTest cases, 454 skips, zero failures, plus 193 Swift Testing cases. Lint, package policy, build, CLI help checks, documentation examples, docs build, and whitespace checks pass.

The earlier locked-device attempt is preserved as failed: it reported 52 failures in protected run-record storage suites while macOS reported `IOConsoleLocked=Yes`. Those failures disappeared in the unlocked rerun without runtime or test changes. Both logs and their hashes remain in the evidence.

All 50 checkpoint source files and all four input-image hashes still match their receipts. Corpus and protocol contents match committed baseline `c615cb30`. No hosted CI or PR publication was run.

## Reproduce

Use the pinned snapshots and frozen corpus from the October 15 report. Select a fresh evidence root with the same parent model/fixture layout, preserve prior exports, and link its `reference-2`, `reference-4`, and `reference-9` directories to the immutable upstream exports. Run the existing native export and comparison commands for each size with `MERERUN_TEST_BONSAI_PARITY_ROOT` set to the fresh root. The new run uses the same `protocol.json` and `cases.json` bytes.

Run the small independent GPU fixture with:

```bash
MERERUN_TEST_MLX_DEVICE=gpu swift test --filter Bonsai2SmallTests
```
