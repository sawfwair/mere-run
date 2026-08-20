# Documentation style audit

This inventory tracks the Google developer documentation style pass across the
61 Markdown files in the public documentation site and its root contributor
entry points: `README.md`, `CODEBASE.md`, and `CONTRIBUTING.md`.

Audit date: August 20, 2026

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
- `README.md`
- `docs/architecture/audio-enhancement-ap-bwe-report.md`
- `docs/architecture/audio-enhancement-universr-report.md`
- `docs/architecture.md`
- `docs/architecture/music-source-separation-roformer-report.md`
- `docs/architecture/vfx-geometry-model-report.md`
- `docs/benchmark-fused.md`
- `docs/benchmarks/minimax-h3-bf16-m4-max.md`
- `docs/benchmarks/minimax-h3-compact-bf16-q8-m4-max.md`
- `docs/benchmarks/minimax-h3-h3c-transfer.md`
- `docs/benchmarks/minimax-h3-ref2va-mlx-8bit.md`
- `docs/benchmarks/vfx-geometry-apple-silicon.md`
- `docs/benchmarks/laguna-min-p-m4-max.md`
- `docs/benchmarks/fused-reference-runs.md`
- `docs/benchmarks/minimax-music3-m4-max.md`
- `docs/benchmarking.md`
- `docs/cli.md`
- `docs/cookbooks.md`
- `docs/configuration.md`
- `docs/development-workflow.md`
- `docs/README.md`
- `docs/documentation-style.md`
- `docs/documentation-style-audit.md`
- `docs/gate.md`
- `docs/graph/studio.md`
- `docs/index.md`
- `docs/internals/cli-and-runtime.md`
- `docs/internals/dit-performance.md`
- `docs/internals/guarded-acceleration.md`
- `docs/internals/source-layout.md`
- `docs/ios-studio.md`
- `docs/ltx25-upstream-parity.md`
- `docs/linux-quickstart.md`
- `docs/macos-deep-links.md`
- `docs/macos-studio-roadmap.md`
- `docs/mlx-swift-fork.md`
- `docs/plugins.md`
- `docs/raycast.md`
- `docs/repository-tour.md`
- `docs/evaluation-packs.md`
- `docs/falcon-perception-disparity-report.md`
- `docs/runtime/acestep-validation.md`
- `docs/getting-started.md`
- `docs/internals/structured-runs-preflight-actions.md`
- `docs/model-sources.md`
- `docs/runtime/api-server.md`
- `docs/runtime/audio.md`
- `docs/runtime/geo.md`
- `docs/runtime/image.md`
- `docs/runtime/model-management.md`
- `docs/runtime/music.md`
- `docs/runtime/sfx.md`
- `docs/runtime/speech.md`
- `docs/runtime/text.md`
- `docs/runtime/video.md`
- `docs/runtime/vision.md`
- `docs/runtime/world.md`
- `docs/testing.md`
- `docs/workflows.md`

The VitePress navigation in `docs/.vitepress/config.mts` also received a full
sentence-case and destination review.

## Remaining pages

None. All 61 files in this audit have received a complete pass.

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
