# Ornith Q4 distilled MTP promotion

The managed `text-agent-ornith-35b-mlx-4bit` bundle now uses Shisa's
Ornith-distilled BF16 draft head. The target, tokenizer, configuration, and
vision files retain their previous hashes. The compact head is 1,689,376,032
bytes, replacing a 4,378,994,104-byte shard that also contained unused target
weights. Q6, Q8, and BF16 companion selection is unchanged.

The [preceding head comparison](ornith-mtp-upstream-check-2026-09-05.md)
established better drafting on its two prompts. With three proposals per round,
code acceptance increased from 34.8% to 70.7% and prose acceptance from 13.4%
to 31.0%. All 40 measured requests retained the target's greedy output hash.
This promotion adds compatibility checks around that result.

## Artifact and provenance

The [published bundle](https://huggingface.co/Sawfwair/Ornith-1.5-35B-A3B-MLX-4bit-Vision-MTP/tree/19a8ee57cb185cb487fa29ff7175c5e1544ebf1c)
is pinned to `19a8ee57cb185cb487fa29ff7175c5e1544ebf1c`. The selected download payload is
25,963,186,364 bytes (about 24.2 GiB). The prior immutable bundle revision,
`2323acfa0fd0a01c452c89991558e6bbd86f0f05`, remains available for rollback.

The replacement source is
[`shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY`](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY/tree/2b19b31bfe1659c6b0d9459ec3cbd87e34a322ef),
revision `2b19b31bfe1659c6b0d9459ec3cbd87e34a322ef`. Its original file SHA-256 is
`73c6e839971fff3c6d78dbcb6a15895bbab340a2898e98aa6943070751de712e`.

The layout conversion splits 19 fused BF16 tensors into 785 indexed tensors.
It copies every tensor value unchanged, including raw-HF zero-centered RMSNorm
weights. The packaged file SHA-256 is
`d1b500efb9ed14e3c855ef87a118c1d4a7de27029ed5f27e7e4fae48a1d34028`.
The bundle includes its Apache-2.0 license, notice of modifications, per-tensor
hashes in `mtp/PROVENANCE.json`, and a complete `SHA256SUMS`. The target and
vision components retain their upstream MIT declaration.

## Compatibility checks

The installed-checkpoint tests run on an Apple M4 Max with 128 GiB unified
memory. They use the exact compact head layout and the same target and vision
bytes as the published bundle.

| Case | Check |
|---|---|
| Long context | Retrieve an exact passphrase from a 34,645-token prompt, replay all cached prompt tokens, and match target-only greedy output. |
| Follow-up | Retrieve the passphrase in a 34,674-token conversation using a cache hit, then confirm the original checkpoint is unchanged. |
| Tools | Parse the requested `lookup_project` call with the exact argument, supply a fixture tool result, and match the target-only follow-up answer. |
| Sampling | Generate 96 tokens with temperature 0.7, top-p 0.9, top-k 20, and min-p 0.05 for seeds 7, 42, and 2026, using both MTP and target-only routes. Replay seed 7 with MTP after a cache hit and require identical output. |
| Thinking | Match both reasoning content and the final greedy answer with thinking enabled. |
| Vision | Load all 333 indexed vision tensors, validate their runtime mapping, and identify a red circle and blue square in the image fixture. |
| Random state | Repeat seeded token and acceptance draws across request streams, task suspension, and intervening global random work; check acceptance probabilities zero and one. |

Tool and image requests intentionally use target-only decoding. Their tests
confirm preserved behavior; this head does not accelerate those routes.
Sampling checks establish request execution, seed replay, and routing. They are
not a statistical distribution-equivalence or downstream quality benchmark.
MTP and target-only sampling consume different random draws, so equal seeds
need not produce identical text across those routes.

## Sampling correction found during qualification

A combined run exposed an existing shared-random-state failure after changing
request modes: MLX attempted to evaluate a lazy key graph belonging to another
thread's GPU stream. The Qwen generation path also did not apply `ChatRequest.seed`,
and MTP acceptance used a separate system random draw.

Generation now creates request-local MLX random state after model loading,
using the supplied seed when present. Token sampling and MTP acceptance share
that state. The acceptance comparison excludes probability zero. The focused
GPU regression and installed checkpoint replay check cover this correction.

An earlier tool assertion incorrectly expected MTP routing despite the existing
tool exclusion. The assertion was corrected to check the intended target-only
route, while retaining exact tool-call and follow-up checks.

## Reproduce and upgrade

After updating mere.run, replace an older managed Q4 bundle with:

```bash
mere.run model pull text-agent-ornith-35b-mlx-4bit --force
```

Existing complete installs are preserved until explicitly replaced. Validation
accepts both the legacy final-shard index and the compact head index.

From a source checkout, set the two model-root variables to the same downloaded
Q4 bundle and run the opt-in checks:

```bash
MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_TEST_Q35_TRANSFER=1 \
MERERUN_TEST_Q35_TRANSFER_MODEL_ID=text-agent-ornith-35b-mlx-4bit \
MERERUN_TEST_Q35_TRANSFER_MODEL_ROOT=/path/to/ornith \
MERERUN_TEST_ORNITH_VISION_4BIT_MODEL_ROOT=/path/to/ornith \
swift test --filter 'Q35SamplingTests|Q35TransferCheckpointTests/testInstalledExtendedContextReplay|Q35TransferCheckpointTests/testInstalledToolsSamplingAndThinking|OrnithVisionCheckpointTests'
```

The [promotion receipt](receipts/ornith-mtp-promotion-2026-09-05.json) retains
outputs, routes, token counts, timings, artifact hashes, and source hashes.
These compatibility runs occurred with other desktop work active. Their timings
are operational evidence, not a controlled speedup measurement. This promotion
does not certify the full advertised 262,144-token context, concurrent serving,
other Ornith precisions, or other inference engines.
