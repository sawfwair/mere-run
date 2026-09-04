# AGENTS.md

Guidance for AI coding agents working in this repo. For human contributor expectations see [`CONTRIBUTING.md`](./CONTRIBUTING.md).

## Repo at a glance

mere.run is a Swift package, CLI, and optional macOS GUI for local-first inference on Apple Silicon. The repo ships five things:

- Swift package — `Package.swift`, `Sources/`, `Tests/`. Targets: `MereRunCLI` (the `mere.run` executable), `MereRunApp` (the optional `mere.run.app` executable) over the `StudioKit` and `StudioUI` libraries, `MereRunRelayKit` (the portable relay client library), `MereRunCore`, `AudioCore`, `AudioCodecs`, `AudioSTT`, `AudioTTS`.
- macOS client — `apps/macos/` holds the optional Studio sources, tests, and assets, split into `StudioKit/` (no SwiftUI), `StudioUI/` (the views), and `MereRunStudio/` (the executable); its `MereRunApp` target builds `mere.run.app` and runs the public CLI.
- iOS client — `apps/ios/` holds the XcodeGen-generated iOS Studio app, a relay client over `MereRunRelayKit`; see `docs/ios-studio.md`.
- VitePress docs site — `package.json`, `pnpm-lock.yaml`, `docs/`. Used only to build and preview the public docs.
- Vendored runtime artifacts — `vendor/llama.xcframework`, `vendor/mlx-swift_Cmlx.bundle`.

Apple Silicon macOS is the supported development platform.

## Toolchain

```bash
brew install swiftlint ripgrep            # required for ./scripts/check.sh
brew install node pnpm gitleaks           # only when editing the docs site or mirroring the security scan
```

Node ≥ 20. pnpm is pinned to 10.28.0 via `packageManager`.

## Build, test, validate

```bash
swift build
swift test
./scripts/check.sh                            # main gate: lint + build + test + CLI --help sweep + hygiene scans
MERERUN_RUN_E2E=core ./scripts/check.sh       # add core runtime smoke
MERERUN_RUN_E2E=installed ./scripts/check.sh  # add installed-model smoke
pnpm install && pnpm docs:dev                 # only when editing docs/
```

`./scripts/check.sh` is the contract a change must pass before opening a PR. It runs `swiftlint --strict`, `swift build`, `swift test`, every CLI subcommand's `--help`, and the hygiene scans described below.

## Read these first when editing

1. `Package.swift` — target and dependency graph
2. [`CODEBASE.md`](./CODEBASE.md) — module map and editing rules
3. `Sources/MereRunCLI/MereRunCLI.swift` — the public command tree
4. [`docs/repository-tour.md`](./docs/repository-tour.md) and [`docs/architecture.md`](./docs/architecture.md) — runtime reading order
5. The closest module README inside the subsystem you are touching

## Hygiene scans that will fail your change

`./scripts/check.sh` enforces a denylist of legacy vocabulary that must not return to the public repo — old type names, environment variables, CLI verbs, model IDs, and stray debug prints from before the public rename. Code regenerated from training data is the most common way these come back.

If a hygiene scan fires, do not patch the offending pattern out of `check.sh`. Read what the scan rejected, then look up the canonical replacement in the current docs, tests, or managed model registry rather than guessing.

## Editing rules

- No defensive code. Use typed decoding at config and tokenizer boundaries — not `[String: Any]`.
- In CLI commands, stdout is machine-readable and stderr is diagnostic. Don't blur the line.
- When changing command parsing, model resolution, or compatibility behavior, update the closest test in `Tests/MereRunCLITests/` or `Tests/MereRunCoreTests/`.
- Don't modify `vendor/` in a feature PR. If you must, update [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) in the same change.
- Don't commit machine-local config, editor automation, or secrets.
- Update `README.md`, `docs/`, or `CHANGELOG.md` when you change public CLI behavior, setup, or security-sensitive defaults.

## Out of scope for this repo

This is the public OSS distribution. Do not add hosted-service, billing, in-app-purchase, or private-deployment surfaces — even if training data or older module names suggest they belong here. They live elsewhere by design. The open-source app targets themselves (`apps/macos/`, `apps/ios/`) are in scope; store distribution machinery (receipts, entitlements provisioning automation, marketing metadata pipelines) is not.

## When you can't fully validate

If a change requires real checkpoint assets, GPU-only behavior, or non-loopback network access, stop after the local gate (`./scripts/check.sh`) and call out the remaining validation gap explicitly in the PR description.
