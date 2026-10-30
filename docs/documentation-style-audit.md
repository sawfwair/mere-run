# Documentation style audit

This inventory tracks the Google developer documentation style pass across the
130 Markdown files in the public documentation site and the root contributor
entry points: `README.md`, `CODEBASE.md`, `CONTRIBUTING.md`, `DECISIONS.md`,
`DESIGN.md`, `PRODUCT.md`, and `SECURITY.md`.

Audit dates: August 20, 2026 (first pass, 61 files) and October 29, 2026
(accuracy and style refresh against v0.67.1, 137 files).

The October pass checked each page against the v0.67.1 command tree,
`mere.run --help`, the managed model catalog, and `CHANGELOG.md`, and brought
the pages added since August through the same style review.

## Status definitions

- **Complete:** Reviewed line by line for audience, active voice, present tense,
  sentence case, terminology, link text, accessibility, example data, and
  time-sensitive claims.
- **Targeted:** Updated for a specific issue or normalized structurally, but
  still requires a complete line-by-line review.
- **Pending:** Not changed during this pass.

Generated command inventories and historical benchmark receipts have additional
constraints. Update generated text through its source, and preserve recorded
measurements, hashes, versions, and limitations when improving historical prose.

## Complete

- `CODEBASE.md`
- `CONTRIBUTING.md`
- `DECISIONS.md`
- `DESIGN.md`
- `PRODUCT.md`
- `README.md`
- `SECURITY.md`
- `docs/README.md`
- `docs/architecture.md`
- `docs/architecture/audio-enhancement-ap-bwe-report.md`
- `docs/architecture/audio-enhancement-universr-report.md`
- `docs/architecture/laya-native-qualification-2026-09-22.md`
- `docs/architecture/music-source-separation-roformer-report.md`
- `docs/architecture/vfx-geometry-model-report.md`
- `docs/benchmark-fused.md`
- `docs/benchmarking.md`
- `docs/benchmarks/auk-native-qualification-2026-09-18.md`
- `docs/benchmarks/bonsai2-small-2026-10-15.md`
- `docs/benchmarks/bonsai2-small-fixed-2026-10-16.md`
- `docs/benchmarks/bonsai2-small-selected-quality-2026-10-16.md`
- `docs/benchmarks/bonsai2-small-throughput-2026-10-16.md`
- `docs/benchmarks/bonsai2-small-upstream-parity-2026-10-15.md`
- `docs/benchmarks/bonsai2-small-upstream-parity-2026-10-16.md`
- `docs/benchmarks/clef-flash-native-qualification-2026-10-02.md`
- `docs/benchmarks/clef-native-qualification-2026-10-01.md`
- `docs/benchmarks/clef-omni-q4-qualification-2026-10-09.md`
- `docs/benchmarks/d1-native-qualification-2026-10-09.md`
- `docs/benchmarks/flash-next-verification-frontier-2026-09-03.md`
- `docs/benchmarks/flux3-klein-native-qualification-2026-10-15.md`
- `docs/benchmarks/fused-reference-runs.md`
- `docs/benchmarks/kolibri-native-qualification-2026-10-04.md`
- `docs/benchmarks/laguna-min-p-m4-max.md`
- `docs/benchmarks/lightonocr3-native-qualification-2026-10-08.md`
- `docs/benchmarks/minimax-h3-bf16-m4-max.md`
- `docs/benchmarks/minimax-h3-compact-bf16-q8-m4-max.md`
- `docs/benchmarks/minimax-h3-fasth3-vsa-datafree-m4-max.md`
- `docs/benchmarks/minimax-h3-h3c-transfer.md`
- `docs/benchmarks/minimax-h3-ref2va-mlx-8bit.md`
- `docs/benchmarks/minimax-music3-m4-max.md`
- `docs/benchmarks/ornith-mtp-promotion-2026-09-05.md`
- `docs/benchmarks/ornith-mtp-upstream-check-2026-09-05.md`
- `docs/benchmarks/ornith-qwen-flash-transfer-2026-09-05.md`
- `docs/benchmarks/ornith-qwen-generation-tuning-2026-09-05.md`
- `docs/benchmarks/qwen-image-21-native-qualification-2026-09-20.md`
- `docs/benchmarks/qwen-image-21-turbo-mixed-qualification-2026-10-09.md`
- `docs/benchmarks/vfx-geometry-apple-silicon.md`
- `docs/business/adapt-support-workflow.md`
- `docs/business/customer-support.md`
- `docs/business/direct-ticket-triage.md`
- `docs/business/index.md`
- `docs/business/shared-archive.md`
- `docs/cli.md`
- `docs/configuration.md`
- `docs/cookbooks.md`
- `docs/development-workflow.md`
- `docs/documentation-style-audit.md`
- `docs/documentation-style.md`
- `docs/evaluation-packs.md`
- `docs/falcon-perception-disparity-report.md`
- `docs/gate.md`
- `docs/getting-started.md`
- `docs/graph/studio.md`
- `docs/index.md`
- `docs/internals/catalog-contracts.md`
- `docs/internals/chat-execution.md`
- `docs/internals/cli-and-runtime.md`
- `docs/internals/decode-runtime-boundaries.md`
- `docs/internals/dit-performance.md`
- `docs/internals/gemma-runtime-boundaries.md`
- `docs/internals/geometry-generation-operation.md`
- `docs/internals/guarded-acceleration.md`
- `docs/internals/h3-laguna-runtime-boundaries.md`
- `docs/internals/image-generation-operation.md`
- `docs/internals/image-runtime-boundaries.md`
- `docs/internals/image-training-execution.md`
- `docs/internals/inference-admission.md`
- `docs/internals/ltx-runtime-boundaries.md`
- `docs/internals/package-policy.md`
- `docs/internals/qwen-runtime-boundaries.md`
- `docs/internals/runtime-residency.md`
- `docs/internals/source-layout.md`
- `docs/internals/speech-runtime-boundaries.md`
- `docs/internals/speech-synthesis-operation.md`
- `docs/internals/speech-transcription-operation.md`
- `docs/internals/structured-runs-preflight-actions.md`
- `docs/internals/studio-prompt-workspace.md`
- `docs/internals/text-training-execution.md`
- `docs/internals/video-generation-operation.md`
- `docs/internals/vision-generation-operations.md`
- `docs/internals/workflow-execution.md`
- `docs/ios-studio.md`
- `docs/linux-quickstart.md`
- `docs/ltx25-upstream-parity.md`
- `docs/macos-deep-links.md`
- `docs/macos-studio-capability-review.md`
- `docs/macos-studio-roadmap.md`
- `docs/macos-studio-v2-completion.md`
- `docs/macos-studio-v2.md`
- `docs/mlx-swift-fork.md`
- `docs/model-sources.md`
- `docs/plugins.md`
- `docs/provider-prompting-guides.md`
- `docs/qualification/drama3-native-20261015/README.md`
- `docs/qualification/nemotron-transcribe-int4-20261028/README.md`
- `docs/qualification/nemotron-transcribe-native-20261028/README.md`
- `docs/qualification/nemotron35-bolt-checkpoint-2026-10-15.md`
- `docs/qualification/pplx-embed-v2-9b-2026-10-09.md`
- `docs/qwen-image-edit-2511-native-research-2026-08-29.md`
- `docs/raycast.md`
- `docs/repository-tour.md`
- `docs/runtime/acestep-validation.md`
- `docs/runtime/api-server.md`
- `docs/runtime/audio.md`
- `docs/runtime/clef.md`
- `docs/runtime/d1.md`
- `docs/runtime/embeddinggemma2.md`
- `docs/runtime/flint.md`
- `docs/runtime/geo.md`
- `docs/runtime/gliner25-decide.md`
- `docs/runtime/image.md`
- `docs/runtime/kolibri.md`
- `docs/runtime/laya.md`
- `docs/runtime/model-management.md`
- `docs/runtime/music.md`
- `docs/runtime/nemotron-bolt.md`
- `docs/runtime/nemotron-transcribe.md`
- `docs/runtime/pplx-embed-v2.md`
- `docs/runtime/sfx.md`
- `docs/runtime/speech.md`
- `docs/runtime/text.md`
- `docs/runtime/video.md`
- `docs/runtime/vision.md`
- `docs/runtime/whistle.md`
- `docs/runtime/world.md`
- `docs/testing.md`
- `docs/workflow-batches.md`
- `docs/workflows.md`

The VitePress navigation in `docs/.vitepress/config.mts` also received a full
sentence-case and destination review.

## Remaining pages

None. All 137 files in this audit have received a complete pass.

## Complete-pass checklist

Move a page to **Complete** only after all of these checks pass:

1. Identify the intended reader and lead with the task or outcome.
2. Use direct, active, present-tense instructions.
3. Use sentence case while preserving proper names.
4. Replace vague links and positional references with descriptive wording.
5. Use reserved values for examples and label real public sources accurately.
6. Replace unstable words such as "latest" with a version, date, or named state
   when possible.
7. Preserve command accuracy, generated-content ownership, benchmark receipts,
   security boundaries, and validation limits.
8. Run `bash ./scripts/check-docs-examples.sh`, `pnpm docs:build`, and the
   relevant documentation contract tests.
