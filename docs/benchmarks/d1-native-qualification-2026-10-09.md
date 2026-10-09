# D1 native checkpoint qualification — 2026-10-09

Both pinned, original LiquidAI safetensors ran through the native Swift/MLX
`text decide` CLI without conversion. D1-3B was tested in BF16 with text and
images; D1 omni-600M was tested in FP32 with text, images, and audio. Each of
five requests contained one yes/no, one choice, and one ordinal score question:
15 decisions total. The runtime computation is from commit `b7a4bde5`.

## Original weight identity

Downloads matched the SHA-256 hashes published by Hugging Face for these exact
revisions. No converted weights, quantization, or Python inference fallback
were used by the native runtime.

| Model | Revision | Weight bytes | SHA-256 |
| --- | --- | ---: | --- |
| D1-3B | `da1fe36a861f24690f27f622dca1d8688503d113` | 6,247,065,504 | `50e03317847caf6df9a9aee27ed40f20554a86a21e60d1d47ba41a422b546c0c` |
| D1 omni-600M | `414f8d6438174f5b2133a9c21a478fc42625e308` | 2,348,774,500 | `0713bb05270c2685ad106522f4092bceeeb3a93cf79b401f399a712296c911e1` |

## Comparison results

The independent reference loads the original checkpoint's Python implementation
and weights on CPU, using BF16 for D1-3B and FP32 for omni. It executes each
question separately to match native execution. Reference generation is test
infrastructure; user inference requires no Python installation.

| Model | Input | Maximum absolute probability difference |
| --- | --- | ---: |
| D1-3B BF16 | Text | 0.001278479 |
| D1-3B BF16 | Image | 0.000642697 |
| Omni FP32 | Text | 0.000001252 |
| Omni FP32 | Image | 0.000008881 |
| Omni FP32 | Audio | 0.000002027 |

Choice identities, yes/no classifications at 0.5, and executed input-token
counts agree for all cases. All CLI commands exited successfully and emitted
valid probability distributions with zero output tokens. Expected score values
agree within the committed numerical tolerances. [Native outputs and comparison
receipt](./d1-native-qualification-2026-10-09.json) retain every answer.

Inputs are in `Tests/MereRunCoreTests/Fixtures/D1/released/`:

- Text: a parcel arrived on time with broken glass inside.
- Image: a locally generated 224×224 red PNG; questions ask about its color and
  red coverage.
- Audio: a 3.5-second, 16 kHz mono PCM clip, synthesized locally with the macOS
  Samantha voice reading the parcel sentence.

Both models select the damage team for text and red for the image. Omni
misanswers two speech questions: it selects book as the damaged item and gives
arrival a probability below 0.5. The original implementation produces the same
answers. The audio result establishes native/reference agreement on this clip,
not speech-task accuracy.

## Reproduction

Place the pinned original checkpoint assets in `causal/` and `omni/` under a
local directory. The native test uses the committed input media and independent
reference answers:

```bash
MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_TEST_D1_RELEASED_CHECKPOINTS=/path/to/original-checkpoints \
  swift test --filter D1ReleasedCheckpointTests
```

The regression checks all probabilities, confidence, expected ordinal scores,
selected choices, answer types, and input/output-token counts. FP32 absolute
tolerance is 0.0001; BF16 absolute tolerance is 0.02. The test skips when the
explicit checkpoint directory is absent. Normal user installation remains
`model pull` followed by `text decide`; the test fixture directory structure is
only for qualification.

To regenerate expected answers, retain the original pinned Python source files
alongside each checkpoint and run `scripts/fixtures/export-d1-release-reference.py`
for each family. The fixture README records exact dependencies and commands.

## Local gates

All 13 focused D1 tests passed with GPU execution, both real checkpoints,
original tokenizers, and complete text/image/audio smoke inputs: zero skips or
failures. `./scripts/check.sh` passed lint, architecture/package-policy checks,
build, 5,349 XCTest cases (443 conditional skips, zero failures), 191 Swift
Testing cases, CLI help, and hygiene scans. The docs site build also passed.

## Scope

This qualifies original-checkpoint loading and agreement for these five
requests. It does not establish benchmark accuracy, latency, maximum-context
behavior, all-input BF16 equivalence, a tiled/multiple-image checkpoint suite,
or a broad audio evaluation. Small independent tensor, resize, and tokenizer
fixtures cover additional component behavior. The upstream tree-packing
optimization remains unimplemented.
