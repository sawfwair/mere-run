# Falcon Perception Disparity Report

Date: 2026-04-12
Repo: `/Users/nerd/mere/run`
Reference: `~/third_party/mlx-vlm/mlx_vlm/models/falcon_perception`

## Goal

Trace the native Swift Falcon Perception path end to end, compare it to the `mlx-vlm`
reference implementation, and identify the concrete disparities that explain why:

```bash
mere.run vision ground ~/Desktop/test.jpeg --query "person"
```

currently returns clearly wrong results:

- `Detections: 55`
- repeated centerline boxes with nearly identical `x`, `h`, and only slightly varying `y`
- output written to:
  - `/Users/nerd/Desktop/test_grounded.jpeg`
  - `/Users/nerd/Desktop/test_grounded.json`

Observed sample from the emitted JSON:

- detection 0: `x=0.50048876`, `y=0.50048876`, `h=0.71263784`, `w=1`
- detection 1: `x=0.56989247`, `y=0.50048876`, `h=0.71263784`, `w=0.038948007`
- detections 2..19: mostly the same narrow centerline box repeated with tiny `y` shifts

This is not “over-detecting people.” It is a malformed generation pattern.

## E2E Trace

### 1. CLI entry

Swift path:

- `/Users/nerd/mere/run/Sources/MereRunCLI/Commands/VisionGroundCommand.swift`

Behavior:

- resolves `vision-ground-falcon-perception` by default
- constructs `FalconPerceptionGrounder`
- calls `ground(imageURL:queries:...)`

Status:

- Matches intended command surface.
- No material disparity found here.

### 2. Model root / resources / tokenizer

Swift paths:

- `/Users/nerd/mere/run/Sources/MereRunCore/FalconPerception/FalconPerceptionGrounder.swift`
- `/Users/nerd/mere/run/Sources/MereRunCore/FalconPerception/FalconPerceptionTokenizer.swift`
- `/Users/nerd/mere/run/Sources/MereRunCore/FalconPerception/FalconPerceptionConfig.swift`

Model root used on this machine:

- `/Users/nerd/Library/Application Support/MereRun/models/vision-ground-falcon-perception`
- symlink target:
  `/Users/nerd/Library/Application Support/MereRun/hub/models/tiiuae/Falcon-Perception`

Validated assets present:

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `model.safetensors`

Confirmed token IDs from `tokenizer.json`:

- `<|image|>` = `227`
- `<|image_cls|>` = `244`
- `<|image_reg_1|>` = `245`
- `<|image_reg_2|>` = `246`
- `<|image_reg_3|>` = `247`
- `<|image_reg_4|>` = `248`
- `<|coord|>` = `240`
- `<|size|>` = `241`
- `<|seg|>` = `262`
- `<|REF_SEG|>` = `258`
- `<|end_of_text|>` = `11`
- `<|start_of_query|>` = `264`
- `<|end_of_query|>` = `263`

Findings:

- The flat Hugging Face Falcon config layout is real; the native config decoder had to be
  fixed to support it.
- The prompt/task token distinction is correct:
  - prompt uses `<|REF_SEG|>` = `258`
  - generated segmentation token is `<|seg|>` = `262`
- Tokenizer/config mismatch is **not** the main source of the bad detections.

Status:

- `config.json` layout bug: fixed
- tokenizer IDs: aligned with reference

### 3. Image preprocessing and prompt construction

Reference path:

- `~/third_party/mlx-vlm/mlx_vlm/models/falcon_perception/processing_falcon_perception.py`

Swift path:

- `/Users/nerd/mere/run/Sources/MereRunCore/FalconPerception/FalconPerceptionProcessor.swift`

Reference behavior:

- resize shortest/longest into `256...1024`
- smart-resize to patch multiples
- normalize with mean/std `0.5`
- prompt:
  `"<|image|>Segment these expressions in the image:<|start_of_query|>{query}<|REF_SEG|>"`
- expand `<|image|>` into:
  - `image_cls`
  - `image_reg_1...4`
  - repeated `img_id` patch tokens
  - `img_end_id`

Swift behavior:

- matches the same resize policy
- matches the same smart-resize policy
- matches the same normalization
- matches the same prompt text
- matches the same image-token expansion strategy

Findings:

- Preprocessing appears aligned.
- No confirmed disparity found here.

### 4. Prefill position handling

Reference paths:

- `~/third_party/mlx-vlm/mlx_vlm/models/falcon_perception/falcon_perception.py`
- `~/third_party/mlx-vlm/mlx_vlm/models/falcon_perception/language.py`

Swift paths:

- `/Users/nerd/mere/run/Sources/MereRunCore/FalconPerception/FalconPerceptionGrounder.swift`
- `/Users/nerd/mere/run/Sources/MereRunCore/FalconPerception/FalconPerceptionModel.swift`

Reference behavior:

- precomputes:
  - `position_ids`
  - `pos_hw`
  - `rope_delta`
  - full Falcon attention mask
- stores them on the language model before prefill
- after prefill:
  - clears `_position_ids`
  - clears `_pos_hw`
  - keeps `_rope_delta`
  - keeps `_full_attn_mask`
- decode then derives positions from `cache_offset + rope_delta`

Previous Swift behavior:

- computed prefill position data
- passed it during prefill
- dropped the decode-time rope/position state on subsequent single-token decode

Current Swift behavior:

- now stores prefill state on the language model
- now clears only prefill-sized tables after prefill
- now derives decode-time positions from `cacheOffset + ropeDelta`

Finding:

- This was a real disparity and a real bug.
- It explains some earlier gibberish/tool-mode degeneration.
- It does **not** explain the remaining 55 repeated person detections by itself.

Status:

- fixed

### 5. Weight loading coverage

Swift path:

- `/Users/nerd/mere/run/Sources/MereRunCore/FalconPerception/FalconPerceptionGrounder.swift`

Checkpoint inspected directly from the `model.safetensors` header.

Confirmed tensor families present:

- `tok_embeddings.weight`
- `img_projector.weight`
- `layers.*.attention.wqkv.weight`
- `layers.*.attention.wo.weight`
- `layers.*.attention.sinks`
- `layers.*.feed_forward.w13.weight`
- `layers.*.feed_forward.w2.weight`
- `coord_encoder.*`
- `coord_decoder.*`
- `size_encoder.*`
- `size_decoder.*`
- `proj_segm.*`
- `conv_segm.*`
- `norm.weight`
- `output.weight`

Important negative result:

- There are **no** serialized Falcon tensors named:
  - `_norm_w_in`
  - `_norm_w_qk`
  - `_norm_w`

Implication:

- The earlier “missing checkpointed norm tensors” hypothesis is false.
- Those weights are runtime defaults in the reference model, not saved tensors.
- The real high-value missing tensor family is `layers.*.attention.sinks`.

### 6. Attention implementation

Reference paths:

- `~/third_party/mlx-vlm/mlx_vlm/models/falcon_perception/language.py`
- `~/third_party/mlx-vlm/mlx_vlm/models/base.py`

Swift path:

- `/Users/nerd/mere/run/Sources/MereRunCore/FalconPerception/FalconPerceptionModel.swift`

Reference behavior:

- each attention layer owns:
  - `self.sinks = mx.zeros((self.n_heads,))`
- checkpoint provides real `layers.*.attention.sinks` tensors
- forward path calls:
  - `scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask, sinks=self.sinks)`
- `mlx-vlm` passes that through to MLX attention with sinks support

Current Swift behavior:

- attention layer has:
  - `wqkv`
  - `wo`
  - RMS-norm weights
- attention layer does **not** have a `sinks` parameter
- checkpoint mapper does **not** map `layers.*.attention.sinks`
- forward path does manual:
  - `scores = q @ k^T * scale`
  - optional boolean masking
  - `softmax(scores)`
  - `weights @ v`
- it does **not** call `MLXFast.scaledDotProductAttention(..., sinks: ...)`

Confirmed disparity:

- The native Swift attention implementation ignores a serialized Falcon tensor family
  that the reference uses on every layer.

Severity:

- High

Reason:

- This is the strongest remaining e2e mismatch between the reference model and the
  native port.
- It affects the core language-model decoding loop, which is exactly where the malformed
  repeated box pattern is appearing.

### 7. Decode loop semantics

Reference path:

- `~/third_party/mlx-vlm/mlx_vlm/models/falcon_perception/falcon_perception.py`

Swift path:

- `/Users/nerd/mere/run/Sources/MereRunCore/FalconPerception/FalconPerceptionGrounder.swift`

Comparison:

- Both implementations:
  - greedy sample `argmax(logits[:, -1, :])`
  - interpret coord/size/seg from the hidden state of the previous step
  - feed the sampled token back through token embedding
  - inject encoded coord/size features when the sampled token is coord/size

Finding:

- Decode-loop structure is broadly aligned.
- The major remaining divergence inside the loop is the attention implementation used to
  produce the next logits.

### 8. Segmentation / visualization path

Swift paths:

- `/Users/nerd/mere/run/Sources/MereRunCore/FalconPerception/FalconPerceptionAnyUp.swift`
- `/Users/nerd/mere/run/Sources/MereRunCore/FalconPerception/FalconPerceptionGrounder.swift`

Status:

- AnyUp and segmentation feature plumbing were ported.
- They are not the best explanation for the current failure because the malformed pattern
  is already visible at the box-generation stage before mask quality matters.

## Fixed Disparities

These were real bugs found during the trace:

1. Flat Hugging Face Falcon config decode
   - Native code originally assumed nested `text_config` / `vision_config`.
   - Real pulled Falcon config is flat.
   - Fixed.

2. Installed/global binary Metal bootstrap
   - Global binary did not reliably colocate/copy MLX metallibs.
   - Fixed.

3. Decode-time rope/position handoff after prefill
   - Native decode was not preserving reference rope state.
   - Fixed.

These were investigated but ruled out as the primary remaining cause:

4. Prompt/tokenizer mismatch
   - Prompt and image-token expansion match reference.
   - Not the primary issue.

5. Missing checkpointed norm tensors
   - Checkpoint header inspection shows those tensors do not exist in the Falcon weights.
   - Not the primary issue.

## Open Disparities

### A. Attention sinks are missing end to end

Evidence:

- checkpoint contains `layers.*.attention.sinks`
- reference passes `sinks=self.sinks` into attention
- native Swift model has no sinks parameter and no sink-aware attention call

Expected fix:

- add `sinks` parameter to `FalconPerceptionAttention`
- map `layers.*.attention.sinks` from checkpoint
- replace manual score/softmax path with `MLXFast.scaledDotProductAttention(..., sinks: sinks)`
  or otherwise replicate the exact MLX sink behavior

Priority:

- P0

### B. Native attention kernel differs from reference MLX attention path

Evidence:

- reference uses `scaled_dot_product_attention(..., cache=cache, sinks=sinks)`
- Swift currently implements attention manually

Risk:

- even if sinks are added, the manual implementation may still diverge from the reference
  in masked/cache cases

Expected fix:

- switch Falcon attention to the same MLX fast SDPA API used elsewhere in this repo

Priority:

- P0

### C. Weight-mapping audit should explicitly report skipped tensor families

Current problem:

- `loadWeights` silently skips tensors whose mapped keys do not match a native parameter
- this made the missing `attention.sinks` family invisible until direct header inspection

Expected fix:

- add a debug/audit mode that records:
  - total tensors
  - mapped tensors
  - skipped tensors
  - skipped tensor families by prefix

Priority:

- P1

## Most Likely Root Cause

The highest-confidence root cause of the remaining bad grounding behavior is:

1. Falcon attention sinks are serialized in the checkpoint.
2. The `mlx-vlm` reference uses them in every attention call.
3. The native Swift attention implementation currently ignores them completely.
4. The malformed output pattern is a generation-time failure, not a prompt-time or
   segmentation-time failure.

That makes the missing sink-aware attention path the most important unresolved e2e
disparity.

## Recommended Next Patch Order

1. Add `attention.sinks` to `FalconPerceptionAttention`.
2. Map `layers.*.attention.sinks` in checkpoint loading.
3. Replace manual attention math with `MLXFast.scaledDotProductAttention`, passing:
   - `queries`
   - `keys`
   - `values`
   - `scale`
   - `mask`
   - `sinks`
4. Re-run:
   - `mere.run vision ground ~/Desktop/test.jpeg --query "person"`
5. Compare:
   - detection count
   - first 10 detection centers/sizes
   - whether the repeated centerline pattern disappears
6. Add an audit command or debug mode that prints skipped checkpoint tensor families.

## Appendix: Repro Evidence

Command:

```bash
mere.run vision ground ~/Desktop/test.jpeg --query "person"
```

Current native result:

- `Model: vision-ground-falcon-perception`
- `Detections: 55`
- `Image: /Users/nerd/Desktop/test_grounded.jpeg`
- `JSON: /Users/nerd/Desktop/test_grounded.json`

Key artifact for inspection:

- `/Users/nerd/Desktop/test_grounded.json`

Symptoms inside JSON:

- repeated boxes centered around `x ~= 0.5005`
- repeated `h ~= 0.7126`
- many boxes differ only by small `y` deltas

This is consistent with a broken decode/attention path, not with genuine multi-person
grounding.
