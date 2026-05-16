# Changelog

All notable changes to this public repository will be documented in this file.

The format is based on Keep a Changelog.

## Unreleased

### Fixed

- fixed DeepSeek V4 Flash model resolution so existing imatrix GGUF symlinks
  are reused instead of triggering another 81 GB download.

## 0.5.2 - 2026-05-15

### Fixed

- fixed `mere.run model pull` so a zero-exit pull is rejected when the model is
  still not discoverable by `mere.run model list`.
- added disk-space preflight checks for managed Hugging Face pulls, including
  actionable cache/model-store relocation guidance for low-space volumes.

## 0.5.1 - 2026-05-15

### Fixed

- fixed the DS4 premier agent tier to pull the page-aligned DeepSeek V4 Flash
  imatrix GGUF and ship the matching refreshed `ds4` runtime binaries.

## 0.5.0 - 2026-05-13

### Added

- added the DS4 premier agent tier for 96 GB+ Apple Silicon Macs, including
  vendored `ds4` runtime packaging, setup readiness checks, and OpenAI-compatible
  local serving through `mere.run agent start`.
- added Pi agent installation and launch support so high-memory Macs can choose
  between the DS4 premier path and the Pi local agent runtime.
- expanded `mere.run api serve` Chat Completions compatibility with typed
  OpenAI request fields, per-engine capability validation, native function-tool
  mapping, and streaming usage chunks.

### Changed

- changed the macOS Studio first-launch behavior so dragging the app into
  Applications no longer auto-installs the terminal CLI; Settings now provides
  explicit buttons to install the bundled CLI and optional `use-mere-run` Codex
  skill.

## 0.4.13 - 2026-05-12

### Fixed

- fixed Image Nano (`image-zimage-nano`) generation from the public mflux Z-Image checkpoint by mapping mflux VAE wrapper keys onto the Swift VAE loader and aligning sigma shift values with mflux.

## 0.4.12 - 2026-05-12

### Added

- added native HiDream O1 image generation support with managed model IDs, validation, CLI reference-image options, tests, docs, and optional e2e smoke coverage.

### Changed

- bumped `github.com/huggingface/swift-transformers` from 1.3.0 to 1.3.2.

## 0.4.11 - 2026-05-11

### Fixed

- fixed public app release bundles so the packaged Studio app embeds a freshly built `mere.run` CLI payload and its runtime support assets from the same checkout.

## 0.4.10 - 2026-05-10

### Added

- added a Studio Models sheet for inspecting managed models, revealing their folders in Finder, opening the model store, and purging downloaded installs.

### Fixed

- fixed Image Nano (`image-zimage-nano`) to pull and load the public 4-bit mflux Z-Image Turbo quant layout.
- tightened Studio readiness handling so unsupported or unknown model states cannot be bypassed, model changes refresh capabilities, and preflight failures no longer leave stale running items.

## 0.4.9 - 2026-05-10

### Fixed

- fixed Studio model pulls so the missing-model overlay is replaced by download progress and readiness is rechecked after the pull completes.

## 0.4.8 - 2026-05-10

### Changed

- made Image Nano (`image-zimage-nano`) the default image model for CLI and Studio first-run workflows.

## 0.4.7 - 2026-05-10

### Changed

- updated Studio readiness so modes respect managed model capability support before offering pulls or runs

## 0.4.6 - 2026-05-09

### Fixed

- fixed app launch from `/Applications` when no custom CLI path is set by making package-root discovery stop reliably at the filesystem root

## 0.4.5 - 2026-05-09

### Fixed

- fixed Studio first-open previews so large media outputs are not read into memory as text and image previews are downsampled before display
- fixed duplicate Studio model-readiness checks from stacking during launch, mode changes, and model changes

## 0.4.4 - 2026-05-09

### Added

- added a site-matched MereRun app icon for the macOS Studio bundle

### Changed

- polished the public DMG into a drag-to-Applications layout with release support files hidden under `.mere-run`

## 0.4.3 - 2026-05-09

### Added

- added `mere.run guide`, an offline CLI cookbook reader with per-command guides across image, text, speech, vision, music, video, model management, API serving, setup, and agent workflows
- added public docs and Codex skills that point agents and users at the same guide material

## 0.4.2 - 2026-05-09

### Added

- added Hugging Face pull sources for `image-klein-nano`, `image-zimage-nano`, and `image-zimage-base`
- added Studio first-launch Terminal CLI bootstrap from the app-bundled payload

### Changed

- removed private archive/R2 model-source downloads; managed pulls now use cataloged Hugging Face snapshots only
- changed the DMG to a Studio-first layout with a separate `CLI/` payload folder and Applications shortcut
- taught `install.sh` to install from the DMG `CLI/` payload folder when present

## 0.4.0 - 2026-05-07

### Added

- third-party notices for vendored runtime artifacts
- GitHub issue templates, pull request template, Dependabot, and a lightweight security workflow
- MereRun Studio, a user-facing SwiftUI macOS app shell with a unified output canvas, mode-aware prompt bar, local Library, guided model readiness, and the technical command surface preserved behind Advanced
- Studio mode mapping for image, text chat/code, speech, vision, music, and video workflows backed by the public `mere.run` CLI
- persistent Studio library metadata under Application Support with local output previews for images, media, text, JSON, and generic files
- repeatable `MereRun.app` bundle creation through `scripts/build_mere_run_app.sh`
- streaming output support for `mere.run text chat --stream`, including live Studio canvas rendering for chat and code runs

### Changed

- clarified public contributor guidance and removed maintainer-specific workflow files from the repo surface
- documented local API serving safety defaults, advanced operator flags, and HTTPS-only remote artifact expectations
- hardened API request validation for generation parameters and operator-controlled LoRA selection
- kept `shell_exec` out of non-interactive tool auto-approval even when `--auto-approve-tools` is passed
- made signed model-source endpoint failures fail closed unless fallback is explicitly allowed
- hardened recoverable runtime construction and conditioning failures to throw typed errors instead of terminating the process
- default GUI launch instructions now use the app bundle path while keeping `swift run mere.run.app` as a contributor smoke path
- Studio CLI resolution now prefers local build products before installed binaries during development

### Fixed

- fixed DMG installs when the optional packaged model-source URL sidecar is absent
- fixed nested ASR managed-model root normalization so qwen3/parakeet alternates do not recurse indefinitely
