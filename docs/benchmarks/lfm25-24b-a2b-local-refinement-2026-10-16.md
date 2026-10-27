# LFM2.5 local mixed-quant refinement

This local refinement does not establish an acceptable advance over its frozen
18.95 GB baseline. The 23.94 GB `mixed6-8-quality` artifact remains the preferred
preserved control. All smaller candidates remain experimental and unqualified.
The broad control in this report's comparisons is the 18.95 GB baseline from
the preceding experiment; it does not replace the larger control. Preserve both
controls, experimental candidates, and negative evidence.

The selected artifact is **18.99578 GB**
(18,995,778,279 bytes), with 68 five-bit and
46 six-bit expert banks. The limit remains 19,000,000,000 bytes.
Its conversion receipt SHA-256 is `fb9749ceb5f23dd536e24a8832eaa76d26a35ba3ee3ca2c6bc97edb1bc1e4bde`.
No model default changes, managed activation, upload, RunPod operation, or Linux
validation occur in this followup.

## Heldout source fidelity

The source is [the pinned BF16 checkpoint](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/tree/1769516ac588c4c558089c94f27d96b0dc67b183),
evaluated locally with FP32 compute and one active layer resident at a time.
The independent implementation uses pinned MLX-LM commit
`ba8a08fd2ed72bbc53de84f89933f91fe4743e4a` and MLX/Metal 0.32.1.
All 10 downloaded source shards pass their pinned SHA-256 checks.

| Heldout measurement | Broad control | Selected local candidate |
| --- | ---: | ---: |
| All-family mean KL | 0.024354898 | 0.024870598 |
| Coding mean KL | 0.0056650548 | 0.0055399545 |
| Tool-use mean KL | 0.0034101221 | 0.0036325714 |
| Coding top-token agreement | 0.98668442 | 0.9840213 |
| Tool-use top-token agreement | 0.98364486 | 0.98364486 |
| Prose mean KL | 0.033812314 | 0.034577501 |
| Top-token agreement | 0.94184436 | 0.93962891 |
| Worst case KL | 0.054565557 | 0.074998655 |

The heldout weighted KL change is
**-1.847% improvement**,
using weights 0.4 coding, 0.4 tool use, and 0.2 prose.
The unchanged heldout aggregate and worst-case source-quality gate
**fails**.
The model retains `quality_qualified: false`.
Top-token agreement measures teacher-forced predictions; it is not a coding
task success rate. Native generated-code correctness remains unscored under
the prior scope amendment.

## Calibration and candidate selection

The preserved broad artifact is the control from the
[previous coding/tool experiment](./lfm25-24b-a2b-coding-tools-2026-10-15.md).
The older small parent and larger control also remain preserved.

Before tuning, the experiment freezes 24 training and 12 reconstruction-selection
rows drawn exclusively from prior calibration data. Coding and tool rows receive
three times the prose weight in local reconstruction. Heldout rows do not enter
weight refinement or candidate selection.

The primary experiment fits bounded FP16 affine coefficients with existing packed
codes and precision policy. It retains a layer update only if the frozen
reconstruction-selection checks pass. It also screens standard six-bit upgrades.
The supplementary experiment tests six-bit half-step grids anchored to the
existing five-bit affine coefficients. Each grid represents the old dequantized
weights exactly, then tests frozen correction fractions toward BF16.
At most two banks gain one bit within the byte cap.

A third experiment directly fits only the final two layers affine coefficients
against BF16-source full-vocabulary KL. It freezes 72 calibration training cases
and 24 internal calibration-validation cases before optimization. Integer codes,
precision allocation, and nonexpert tensors remain unchanged.

Full-vocabulary evaluation of the 96 canonical calibration cases selects
the final candidate before any heldout comparison. Selection requires lower
coding/tool-weighted KL and no family mean KL more than 1% worse than the control.
The 56 heldout cases are reused frozen diagnostics, rather than a fresh blind
benchmark. All original failed or rejected proposals remain preserved.

A temporary filename bug interrupts the first packaging attempt. The corrected
writer replays the unchanged prefix, verifies its saved decisions, and completes
the remaining work. Complete six-bit trial files permit separate file-only
packaging recovery. Original failure receipts remain intact; recovered artifacts
receive new receipts and full audits.

## Native implementation and local execution

All 152 cases and 8,673 scored positions complete in
the native Swift checkpoint test. Against the independent same-quant local
reference, mean KL is `1.13608426e-07`, worst case KL is
`1.156276117e-05`, and top-token agreement is
`100.000000%`.
The unchanged backend gate **fails**
(mean KL at most `1e-6`, every case at most `1e-5`, agreement at least `0.9999`).
The local BF16 reference versus saved CUDA source has mean KL
`0.000163818226` and top-token agreement
`99.942350%`.
This comparison is recorded separately from quantization distortion.

Paired native tool generation was not advanced because the heldout nonregression requirement was unmet.

Release context probes were not advanced for this candidate.

Every model computation uses the shared capacity-one inference lease and
unchanged 16 GiB admission floor. No user apps, other inference jobs, memory
limits, or security settings change. Production runtime and tokenizer sources
remain unchanged from the
[native routing followup](./lfm25-24b-a2b-native-routing-2026-10-16.md).
The October 16 repository check passed `./scripts/check.sh`: 5,454 XCTest
cases (453 skips, zero failures) and 194 Swift Testing cases across seven suites.
The machine-readable receipt records the command, exit code, log hash, and
router source hash. This software result is separate from model quality and
supersedes the earlier locked-device software failure. Hosted CI was not run
for this experiment.

Final source-preservation audits verify all 115 shards, all 656 tensors,
finite affine metadata, the complete precision policy, and bit-exact preservation
of all 314 nonexpert tensors. Control shard hashes also pass after the experiment.
The [machine-readable receipt](./receipts/lfm25-local-refinement-2026-10-16.json)
records the separate quality, backend, tool, context, and preservation results.
Private source weights, frozen fixtures, full logits, failed attempts, executed
tools, original controls, and manifests remain on T7.
