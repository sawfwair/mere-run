# Laya native qualification — September 22, 2026

The native Swift/MLX implementation was compared with the pinned Laya SDK and
all three published checkpoints on an Apple M4 Max with 128 GB unified memory.
The implementation covers managed installation, `text decide`,
`POST /v1/text/decisions`, and the macOS Studio **Text > Decisions** workspace.

## Scope and provenance

- Model repository: `convaiinnovations/laya` at
  `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
- Reference SDK: `NandhaKishorM/laya` at
  `573e5b62696ba441230cd6be71d593331b5d23af`.
- Encoder reference: Transformers 5.0.0, PyTorch CPU evaluation in float32.
- Native evaluation: source-built debug executable, Swift 6.3, macOS 26.5.2,
  MLX Metal, float32 computation from the original checkpoint tensors.
- Base revision: `0093ebb4a63e18e23307fe58213754c0dea5cff6`.
- [Artifact pins](./laya-artifact-pins.json) record checkpoint file sizes,
  repository blobs, and weight SHA-256 digests.
- [Machine-readable results](./laya-native-qualification-2026-09-22.json)
  record the qualified binary digest, source digests, individual cases, raw
  tensor errors, API observations, and verification results.

The typed-decisions checkpoint's actual encoder configuration is English
ModernBERT-large. The multilingual checkpoint uses mmBERT-base. Explicit model
selection follows these configurations. No language router runs implicitly.

## Numerical qualification

The reference replay covers five requests for each checkpoint: English typed
questions, mixed Khmer/Spanish/Hindi text and emoji, a single criterion,
twelve criteria with mask-token sanitization and temperature clamping, and a
long state that requires truncation. Together these contain 33 questions.

All 15 requests matched reference choices and input token counts. Maximum error
across probabilities, scores, confidence, and action probabilities was
`4.893e-5`, within the `2e-4` gate. The SDK rounds public values to four decimal
places; native output retains full precision. Tokenizer checks compare all 33
token sequences and marker positions exactly, rather than only their lengths.

Raw tensor checks use the three-question English batch for each real checkpoint:

| Checkpoint | Maximum decision-logit absolute error | Maximum action-logit absolute error | Maximum action-logit relative error |
| --- | ---: | ---: | ---: |
| English | 5.007e-6 | 0.004639 | 1.326e-6 |
| Multilingual | 4.292e-6 | 0.0006104 | 3.942e-7 |
| Typed decisions | 5.603e-6 | 0.001465 | 3.060e-7 |

Decision logits must be within `2e-5`. Action logits reach thousands, so their
gate bounds `abs(native-reference) / max(abs(reference), 1)` below `2e-6`.
The absolute errors are retained above to make this tolerance explicit.

The seeded synthetic fixture separately exercises every encoder layer,
global/local attention, unequal padding, all question types, and a single
criterion on MLX CPU and Metal. It uses an absolute `2e-5` gate for both heads.
Negative cases cover configuration, tensor names/shapes, nonfinite results,
request structure, and token-budget overflow.

## Application qualification

Managed pulls completed for all three model IDs. Their checkpoint directories
passed the shared install validator, and actual CLI/API inference loaded those
managed files. Install regression coverage includes each subfolder layout and
a missing-weight failure.

The authenticated loopback API completed two identical requests per checkpoint
and two concurrent requests. Runtime status showed three loads, two model
replacements, eight completed requests, and no active or queued work afterward.
Repeated requests reused each resident model. Model discovery exposed all three
installed IDs with the `text.decisions` task.

The route returned 401 for missing authentication, 415 for the wrong content
type, and 400 for malformed JSON, an unrelated model, duplicate question IDs,
an impossible token budget, and an oversized body. The rejected token-budget
request is the single expected failure in the resident's request counter.

Studio command/default-draft contracts and four offscreen workspace renders
cover 768- and 1440-point widths in light and dark appearances. These renders
use the repository's isolated UI fixture. Interactive file-picker and keyboard
accessibility behavior were not independently qualified.

## Reproduce

`./scripts/check.sh` passed 4,546 XCTest cases with 346 skips and zero failures,
plus 51 Swift Testing checks. Asset-dependent Laya tests were then run
explicitly with the pinned checkpoints; the default gate skips those tests
when their environment variables are absent.

Run the repository gate and build the documentation:

```bash
./scripts/check.sh
pnpm install --frozen-lockfile
pnpm docs:build
git diff --check
```

Use separately downloaded, pinned checkpoints with the reference tooling. The
checkpoint root contains the English model and the `multilingual` and
`typed-decisions` subdirectories. Install PyTorch, Transformers 5.0.0,
safetensors, and NumPy in an isolated Python environment. Python is used only
for reference generation and comparison.

```bash
python scripts/laya-qualify.py \
  --upstream /path/to/pinned-laya-sdk \
  --checkpoints /path/to/checkpoints \
  --native-bin "$PWD/.build/debug/mere.run" \
  --output /tmp/laya-qualification

MERERUN_LAYA_CHECKPOINTS=/path/to/checkpoints \
MERERUN_LAYA_REFERENCE=/tmp/laya-qualification \
MERERUN_LAYA_PARITY_OUTPUT=/tmp/laya-raw-parity.json \
MERERUN_TEST_MLX_DEVICE=gpu \
swift test --filter LayaTests

MERERUN_STUDIO_SNAPSHOT_DIR=/tmp/laya-studio \
swift test --filter StudioSnapshotTests/testLayaDecisionWorkspaceSnapshots
```

Keep the executable unchanged during a reference replay. The script records
its SHA-256 before evaluation and rejects a run if the executable changes.
Use [the runtime guide](../runtime/laya.md) for request examples and
[the API guide](../runtime/api-server.md) for serving.

## Evidence limits

This qualification establishes the tested implementation's numerical and
integration behavior. It does not measure domain accuracy or calibrate a
decision policy. Both English checkpoints clamp the upstream `choice:11+`
temperature; action probabilities saturated at 1.0 in the tested cases.
Validate confidence thresholds on held-out data for the intended task.

CLI wall times were observed on a shared development machine and are excluded
from performance claims. Minimum-memory hosts, extended-context stress,
Linux/CUDA, iOS, signed application packaging, CI, and release distribution
were not qualified by this local run.
