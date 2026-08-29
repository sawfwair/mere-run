# Qwen Image Edit 2511 native implementation research

This note defines a native Swift and MLX implementation target for Qwen Image
Edit 2511. It separates reference behavior, proposed product behavior, and
unverified performance work.

The source review was completed on August 29, 2026. No model inference or
Apple Silicon benchmark was run for this research. Any performance or quality
claim attributed to an upstream project remains upstream-reported until a
mere.run qualification receipt reproduces it.

## Implementation status

The `codex/qwen-edit-2511-h3-lightx2v` branch now contains an implementation
candidate for the reference path described here. It adds the separate managed
IDs `image-qwen-edit-2511` and `image-qwen-edit-2511-lightning`, immutable base
and adapter sources, ordered one-to-three-reference CLI and API routing,
independent reference geometry, the official prompt/mask boundary, packed
appearance tokens, exact Qwen2.5-VL merged-patch ordering, gated vision MLPs,
multimodal position IDs, per-channel VAE normalization, shape-aware three-axis
transformer RoPE, `zero_cond_t`, the shifted scheduler, positive-norm true CFG,
and the pinned four-step rank-64/alpha-8 Lightning adapter. The 2511 reference
lanes retain BF16 encoder and transformer weights rather than silently applying
the legacy Q4 conversion.

Local tensor and contract tests exercise those boundaries. The loaders now
require exact encoder, transformer, and VAE checkpoint-key coverage before
applying weights, and the managed Lightning adapter is pinned by byte count and
SHA-256. This is still an implementation candidate, not a qualified model
release: no 57.7 GB real checkpoint was downloaded, no real-checkpoint weight
application or tensor comparison was run, no output was visually reviewed, and
no Apple Silicon memory, quality, or timing receipt exists yet.

## Decision

Build a new, reference-matched Qwen Image Edit 2511 path. Don't incrementally
extend the current `QwenImageEditGenerator` denoising core.

The current path is not only missing multi-reference input. It differs from the
official 2511 pipeline at the VAE normalization, prompt template, token layout,
transformer modulation, rotary positions, scheduler, and classifier-free
guidance boundaries. Adding more file paths or a Lightning LoRA before fixing
those boundaries would make an unsupported implementation faster.

Use the new managed model ID `image-qwen-edit-2511`, and preserve
`qwen-image-edit` as a legacy identity until the new path passes the full gate.
Do not silently repoint an installed legacy model to different weights.

Ship two modes first:

| Mode | Model and schedule | Initial status |
|---|---|---|
| Fast | `image-qwen-edit-2511-lightning`: pinned 2511 base plus the pinned four-step BF16 Lightning adapter, four steps, true CFG disabled | Implemented candidate; hardware qualification pending |
| Quality | `image-qwen-edit-2511`: pinned 2511 base, 40 steps, blank negative prompt, true CFG `4.0` | Implemented reference candidate; real-checkpoint qualification pending |

An eight-step 2511 BF16 adapter exists in the pinned Lightning repository, but
its model card still documents the four-step files as the supported core set.
Treat eight steps as a balanced-mode candidate, not as a public recipe, until
its exact adapter, schedule, and quality behavior are qualified. Do not use an
unadapted eight-step base-model run as the medium tier.

## Pinned sources

| Source | Revision | Use |
|---|---|---|
| [Qwen Image Edit 2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511/commit/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9) | `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9` | Model configuration, weights, processor configuration, license, and official recipe |
| [Diffusers 2511 support](https://github.com/huggingface/diffusers/commit/b8a4cbac14d32afa6c6e6c5b9cd17f9715214220) | `b8a4cbac14d32afa6c6e6c5b9cd17f9715214220` | Reference preprocessing, prompt encoding, transformer, scheduler, and CFG math |
| [2511 Lightning artifacts](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/tree/d74eba145674fd7e31b949324e148e21e7118abd) | `d74eba145674fd7e31b949324e148e21e7118abd` | Distilled adapter candidates and scaled FP8 artifacts |
| [mflux 0.18.0](https://github.com/filipstrand/mflux/commit/48e5ae662c003db697af733f677885739483ff28) | `48e5ae662c003db697af733f677885739483ff28` | Apple and MLX cross-check for the earlier 2509 pipeline |
| mere.run | `d9cf81a1622bba18fefbfcb472b2f5645f8d48db` | Native comparison baseline, release 0.46.0 |

The 2511 model and Lightning repositories declare Apache-2.0. Artifact
publication is not enough for a managed pack: the conversion receipt must also
record every source revision, file digest, selected tensor, quantization rule,
and adapter transform.

The pinned Lightning repository contains these two BF16 adapter candidates:

| Artifact | SHA-256 | Size |
|---|---|---:|
| `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | `22226e8d05d354bb356627d428809f5afd7819399b077238a2b70a82883a904f` | 849,608,296 bytes |
| `Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors` | `a9e81a58a78f260f67b337a6f615e8fa4cd3bc79847c77b7d61a581b789b1ba8` | 849,608,296 bytes |

The LightX2V project documents grid artifacts when a BF16-trained Lightning
LoRA is combined with a naively downcast, unscaled FP8 base. Don't copy that
combination into MLX. A quantized managed pack and its adapter must be tested as
one artifact contract.

## Official 2511 execution contract

The model index selects `QwenImageEditPlusPipeline`. The transformer has 60
layers, 24 attention heads, a head dimension of 128, `axes_dims_rope` of
`[16, 56, 56]`, 64 input channels, 16 output latent channels, and
`zero_cond_t: true`. It does not use a distilled guidance embedding.

### Reference preprocessing

Every ordered reference has two independent representations:

- Qwen2.5-VL receives an aspect-preserving image near 147,456 pixels
  (`384 * 384`).
- The VAE receives an aspect-preserving image near 1,048,576 pixels
  (`1024 * 1024`).
- Each width and height is rounded to a multiple of 32.
- Each reference keeps its own aspect ratio and therefore its own semantic
  grid, VAE latent shape, token length, and rotary coordinates.

Output size is a separate choice. The reference pipeline defaults to an area of
1,048,576 pixels using the last reference's aspect ratio. Mere.run should not
hide that policy in list ordering. Use explicit output dimensions when
provided; otherwise, use the primary reference's aspect ratio and record that
decision in the run receipt.

The public gate should initially accept one to three ordered references. The
model pipeline isn't the source of this limit; the limit makes admission,
memory use, and product behavior predictable. Preserve the order across the
CLI, API, app, caches, prompt labels, and receipt.

### Prompt and semantic conditioning

The official prompt template asks the Qwen2.5-VL encoder to describe key image
features and explain the requested alteration while retaining appropriate
consistency. It formats references as `Picture 1`, `Picture 2`, and so on,
using one vision placeholder per reference inside a Qwen chat conversation.

The processor expands each placeholder according to that reference's vision
grid. After Qwen2.5-VL encoding, the pipeline applies the attention mask and
drops the first 64 template tokens. Both the embeddings and the resulting mask
are transformer inputs. The negative CFG pass uses the same ordered reference
images and a separate prompt encoding.

The implementation should use a typed prompt-template revision. A template
change invalidates semantic caches and requires a parity-fixture update.

### Appearance conditioning and latent layout

The VAE takes the deterministic latent mode for each reference. It normalizes
all 16 latent channels using the checkpoint's `latents_mean` and
`latents_std`, then packs two-by-two spatial latent patches. References are
encoded independently and concatenated along the image-token sequence.

The generated image starts from pure random packed latents. The reference
latents are appended to the transformer image stream at every denoising step;
they aren't blended into the output noise. The transformer receives one shape
record for the output followed by one shape record for each ordered reference.

### Transformer behavior

The transformer projects the complete `[output, reference 1, reference 2, ...]`
image-token sequence through `img_in`. It applies `txt_norm` before `txt_in`,
uses non-affine block LayerNorm, and computes three-axis rotary embeddings from
the complete list of image shapes and the text mask lengths.

The 2511-specific `zero_cond_t` flag is a correctness requirement. The
transformer makes two timestep embeddings: the current denoising timestep for
output tokens and timestep zero for all reference tokens. Every block selects
the appropriate modulation per token. The final output discards reference
tokens and returns only the generated image tokens.

Block modulation is `x * (1 + scale) + shift`. Final adaptive normalization
uses the same scale convention before the output projection.

### Scheduler and true CFG

For `N` inference steps, the pipeline supplies `N` raw sigmas linearly spaced
from `1.0` through `1/N`. It computes a resolution-dependent exponential
shift, stretches the final shifted sigma to `0.02`, and appends terminal sigma
zero for the Euler updates. The shifted sigmas multiplied by 1,000 are the
model timesteps.

For a 1024 x 1024 output, the official four-step scheduler timesteps are about
`[1000.000, 766.709, 455.614, 20.000]`. The transformer receives those values
normalized by 1,000: `[1.000000, 0.766709, 0.455614, 0.020000]`. The baseline
mere.run implementation feeds an unshifted schedule while stepping over a
separately shifted sigma schedule. This is a reference-visible mismatch.

The quality recipe uses traditional CFG, called `true_cfg_scale` upstream. It
runs positive and negative transformer passes, computes
`negative + scale * (positive - negative)`, and rescales each combined token to
the positive prediction's norm. A blank negative prompt such as one space is
intentional because the pipeline only activates true CFG when a negative
prompt is present.

## Baseline mere.run gaps

The following release blockers describe the pinned `d9cf81a1` baseline, not
the implementation candidate summarized above. They are ordered by dependency
rather than by estimated effort:

| Area | Current behavior | Required 2511 behavior |
|---|---|---|
| Model identity | Resolves `Qwen/Qwen-Image-Edit` at mutable `main` | New managed 2511 ID with immutable source and converted artifact revisions |
| Request | Requires one `inputImage`; ignores `referenceImages` | One to three ordered references shared by CLI, API, and app |
| Reference resize | Stretches the input to the requested output size | Independent aspect-preserving 384-squared semantic and 1024-squared appearance preprocessing per reference |
| VAE normalization | Loads per-channel mean and standard deviation but uses a scalar fallback factor | Use checkpoint `latents_mean` and `latents_std` in encode and decode |
| Output initialization | Blends the source latent with output noise | Pure random output latents; references are appended conditioning tokens |
| Prompt | Repeats one image token placeholder before the raw instruction | Official chat template, ordered `Picture N` labels, per-image placeholder expansion, 64-token drop, and mask |
| Transformer token stream | Embeds appearance latents and appends them to semantic context | Append packed references to the image-token stream before `img_in` |
| Transformer modules | Has no effective `txt_norm`; full-precision loading permits unused keys | Match every checkpoint module and reject unused or missing keys |
| Norm and modulation | Uses affine block norms and final `scale * x + shift` | Non-affine block norms and `x * (1 + scale) + shift` |
| Positioning | Uses output-only 2D image RoPE and one-dimensional context RoPE | Three-axis, shape-aware RoPE across output and every reference plus masked text lengths |
| 2511 conditioning | Config doesn't model `zero_cond_t` | Per-token current or zero timestep modulation in every block |
| Scheduler | Raw sigmas end at `1/1000`; model sees unshifted timesteps | Raw sigmas end at `1/N`; model sees shifted and terminal-stretched timesteps |
| CFG | Linear positive and negative combination | Official positive-norm rescale, with serial and memory-admitted batched execution |
| Quantization | Automatically converts full-precision encoder and transformer weights to Q4 | BF16 reference lane, then qualified Q8 default; lower precision remains experimental until paired quality evidence passes |
| Tests | Repository resolution and generic CFG arithmetic | Tensor parity, strict checkpoint coverage, real-checkpoint smoke, visual task matrix, and performance receipts |

The current VAE and scheduler mismatches affect both conditioning and output,
so a plausible image is not sufficient evidence that the pipeline is correct.

## Reference-parity boundary

Product tiers, multi-image input, and approximate caches do not establish 2511
reference parity. A conforming implementation must:

- Match the official 2511 transformer instead of only loading the 2511 files.
- Preserve every reference's own aspect ratio and token geometry.
- Pair each speed mode with the adapter and schedule it was distilled for.
- Make the full 40-step path the quality control.
- Record exact artifacts, reference order, preprocessing, schedule, CFG mode,
  and approximation decisions in every run receipt.

Treat any first-block residual cache as an approximate mode until matched-output
qualification demonstrates its error and performance bounds. It is not an
exact 2511 optimization by construction.

## Proposed native architecture

### Typed request and receipt

Keep `GenerationRequest.inputImage` as the primary edit source and append the
existing ordered `referenceImages`. Resolve both into a typed internal list:

```text
QwenEditReference(index, role, url, contentDigest)
```

The initial roles are `primary` for the first image and `reference` for later
images. A future app can expose subject, object, or style labels without
changing the checkpoint prompt contract; the upstream prompt still uses the
deterministic `Picture N` ordering.

Add an immutable execution receipt that includes:

- base repository and commit;
- managed artifact ID, revision, and quantization layout;
- adapter repository, commit, file, SHA-256, rank, alpha, and scale;
- ordered reference content digests and roles;
- semantic and VAE dimensions for every reference;
- output-size policy and final dimensions;
- prompt-template and processor revisions;
- raw sigmas, shifted sigmas, model timesteps, step count, and seed;
- true CFG scale, negative-prompt digest, and serial or batched execution;
- exact cache hits, approximate cache decisions, and component timing;
- peak active, cache, and process memory where available.

### Conditioning plan

Create a `QwenImageEditConditioningPlan` before loading large model components.
It validates reference count and images, calculates every shape, constructs the
ordered prompt placeholders, determines output size, selects the schedule, and
produces all cache keys. No downstream component should infer geometry again.

Represent each reference as separate semantic pixels, semantic grid metadata,
normalized VAE latents, packed appearance tokens, and image shape. Concatenate
only at the same boundaries as the reference pipeline.

### Component lifecycle

Support two explicit residency policies:

- Memory-saving: encode references and prompts, unload the text encoder, load
  the transformer for denoising, then unload it before VAE decode.
- Resident: keep qualified components available across requests on machines
  with measured headroom.

The policy belongs in admission and receipts. It must not silently change
model math.

### Model tiers

Start with a BF16 development pack for parity. Produce a deterministic Q8
managed transformer and text-encoder pack only after BF16 tensor and image
gates pass. Evaluate Q6 as a separate balanced artifact; don't make it the
default based on file size alone. Keep Q4 and unscaled FP8 out of the public
2511 contract.

LoRA application must preserve logical input and output dimensions for
quantized dense layers. Conversion should either merge the adapter before
quantization or prove runtime application against the BF16 reference. Record
the chosen operation in the artifact manifest.

## Optimization order

Implement exact reuse before approximate transformer skipping:

1. Cache deterministic reference VAE latents by model revision, image digest,
   VAE preprocessing dimensions, and normalization revision.
2. Cache complete positive and negative semantic embeddings by ordered image
   digests, prompt digest, template revision, processor revision, and semantic
   dimensions.
3. If the encoder split permits it, cache vision-tower features separately and
   prove that reusing them produces identical prompt embeddings.
4. Generate multiple seeds from one conditioning plan. Batch equal-shape
   candidates when memory admission allows and otherwise process them in
   chunks without re-encoding references.
5. Reuse compiled transformer blocks and qualified Metal attention kernels.
6. Select serial or batched true CFG from measured headroom. Both paths must
   produce the same normalized CFG prediction within tolerance.
7. Add VAE tiling only with overlap and seam tests.

After the exact baseline is qualified, evaluate an opt-in approximate cache.
Use separate positive and negative cache state, include all output and
reference shapes in the key, force full computation during calibration and at
periodic refresh steps, and emit per-step drift and reuse telemetry. Four-step
Lightning is unlikely to have enough steps to benefit and should default to no
approximate cache.

Diffusers now exposes FirstBlock Cache for Qwen Image and other general cache
strategies. Their existence is a research lead, not a portable speed or quality
claim for Apple Silicon or 2511 editing.

## Qualification plan

### Tensor parity

Add a pinned Python exporter under `scripts/reference_parity/` and small checked
fixtures under `Tests/Fixtures/`. Use supplied latents and deterministic image
inputs so random-number-generator differences do not mask model differences.

The fixture set must cover:

- one square reference;
- two references with different aspect ratios;
- three references;
- the same references in reversed order;
- positive and blank-negative semantic encoding;
- four-step and 40-step schedules;
- one CFG combination with positive-norm rescaling.

Compare these boundaries:

1. semantic and VAE resize dimensions;
2. token IDs, image grids, template-drop boundary, and attention masks;
3. Qwen2.5-VL embeddings after mask selection;
4. raw VAE means, normalized latents, and packed reference tokens;
5. output and reference shape records, rotary cosines and sines, and modulation
   indices;
6. transformer input projection, blocks 0 and 59, final normalization, and
   generated-token output slice;
7. raw sigmas, shifted sigmas, model timesteps, and Euler updates;
8. serial and batched true CFG output;
9. final packed latents and decoded pixels for a supplied-noise small case.

Every checkpoint loader must use strict unused-key and shape checks. Add an
expected-key manifest so a missing module cannot pass because the loader
discarded its tensors.

### Real-checkpoint gate

On a machine with the pinned assets, run a low-resolution, one-step BF16 check
against Diffusers at the pinned commit. Then run the complete 40-step control.
Record all component revisions and tensor tolerances. This gate proves runtime
compatibility; it is separate from subjective image quality.

### Visual task matrix

Use paired source images, prompts, seeds, and output dimensions for:

- a localized color or material edit with unaffected-region preservation;
- English and Chinese text replacement;
- single-person identity and pose preservation;
- two-person identity composition;
- product replacement from a second reference;
- industrial material replacement;
- geometric or viewpoint editing;
- one, two, and three references;
- swapped reference order;
- four-step fast and 40-step quality modes.

Record exact outputs and human review separately. Image similarity or VLM
scores are model candidates, not human approval.

### Performance and admission

Measure cold and warm execution separately on physical Apple Silicon machines.
Break out model resolution, each component load, semantic encoding, VAE encode,
transformer denoising, VAE decode, and output write. Record active memory,
cache memory, process peak, swap, candidate batch size, reference count and
shapes, quantization, CFG mode, and cache decisions.

Qualify 48 GB, 64 GB, 96 GB, and larger classes independently where hardware is
available. A successful run on a larger machine doesn't lower the managed
catalog minimum for a smaller class.

Compare alternative implementations only under matched base commit, adapter
hash, precision, reference order and sizes, output dimensions, schedule, CFG,
seed, candidate count, decode accounting, machine, and cold or warm state.

## Delivery sequence

1. Add the pinned Diffusers fixture exporter and failing Swift parity tests.
2. Replace VAE normalization and scheduler behavior.
3. Implement the official prompt processor output, masks, and multi-reference
   conditioning plan.
4. Replace the transformer with strict 2511 module and tensor parity,
   including `txt_norm`, shape-aware rotary positions, and `zero_cond_t`.
5. Qualify one-reference BF16 40-step output.
6. Add two and three references and reference-order receipts.
7. Add strict managed 2511 artifacts and Q8 qualification.
8. Add the pinned four-step Lightning mode and adapter parity.
9. Route the new managed model through `image generate`, the image-edit API,
   and the macOS app with the shared request.
10. Add exact conditioning reuse and candidate batching.
11. Evaluate the eight-step adapter and approximate cache as independent,
    opt-in experiments.
12. Change aliases or defaults only after the new public matrix passes.

## Release gate

Do not describe the implementation as Qwen Image Edit 2511-ready until all of
the following are true:

- the complete reference tensor fixture set passes;
- a pinned real-checkpoint run passes strict load coverage and one-step parity;
- the 40-step BF16 control completes with an immutable receipt;
- one-, two-, and three-reference visual cases are reviewed;
- the four-step adapter is tied to its exact digest and schedule;
- the chosen Q8 pack passes paired BF16 quality and memory gates;
- CLI, API, and app preserve the same ordered-reference semantics;
- a physical-machine admission receipt supports every published memory class;
- any approximate mode is labeled and has paired quality plus timing evidence.

Until then, the work is an implementation candidate, not a qualified model
release.
