# Development Workflow

This page describes the expected local workflow for editing `mere-run`.

Before using `./scripts/check.sh`, install SwiftLint and ripgrep once:

```bash
brew install swiftlint ripgrep
```

For local docs and security checks, install Node.js, pnpm, and Gitleaks:

```bash
brew install node pnpm gitleaks
```

## The normal loop

For most changes:

1. make the code or docs change
2. run the smallest relevant local check
3. run `./scripts/check.sh`
4. run the installed smoke sweep if you touched real runtime behavior

That keeps the package healthy without forcing a full manual validation cycle
for every tiny edit.

## Which command to run

### Docs-only change

```bash
./scripts/check.sh
```

This catches docs hygiene regressions and verifies the CLI help surface still
matches the public tree.

### Command parsing or CLI UX change

```bash
swift test
./scripts/check.sh
```

### Runtime or model-resolution change

```bash
swift test
MERERUN_RUN_E2E=core ./scripts/check.sh
```

If the change affects installed-model behavior, also run:

```bash
MERERUN_RUN_E2E=installed ./scripts/check.sh
```

## Model-store expectations

The public runtime is hard-cut to the canonical OSS model IDs. That means:

- runtime code should use canonical public IDs only
- examples should use canonical public IDs only
- model-store and server troubleshooting should point readers at
  `mere.run status`, `mere.run model list`, `mere.run model info`, and
  `mere.run model repair-manifests`

When testing locally, inspect your current state with:

```bash
swift run mere.run status
swift run mere.run model list
swift run mere.run model info image-klein-max
```

## Editing guidance

### If you touch the public command tree

- keep the modality-first structure intact
- preserve stdout and stderr discipline
- update `docs/cli.md` if the user-facing behavior changes
- update quickstarts, cookbooks, and runtime docs when the command becomes part
  of setup or troubleshooting
- add or update `Tests/MereRunCLITests`

### If you touch runtime code

- keep debug output behind the internal debug helper
- avoid introducing new implicit hosted defaults
- keep model resolution explicit and canonical
- add or update the most local tests you can

### If you touch docs

- teach the current public OSS surface
- prefer links between docs pages over repeating large blocks of reference text

## Contribution boundaries

This repo is intentionally scoped to the package and CLI. Changes should not
reintroduce:

- hosted-service assumptions
- billing or entitlement logic
- app-store-only release behavior
- relay/web/app-specific product layers

See the root `CONTRIBUTING.md` file for the short policy version.

## A good contributor checklist

Before opening a PR:

- `swift build` passes
- `swift test` passes
- `./scripts/check.sh` passes
- `pnpm docs:build` passes if you changed docs
- `gitleaks detect --source . --config .gitleaks.toml --redact --no-banner`
  passes before a public release
- the docs reflect any user-facing change
- the public command and model vocabulary remain canonical

If you changed runtime behavior against real models, also include the result of:

```bash
./scripts/e2e_smoke.sh --core
```

and, when relevant:

```bash
./scripts/e2e_smoke.sh --installed
```
