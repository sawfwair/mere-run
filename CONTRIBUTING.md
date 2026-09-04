# Contributing

- Keep changes scoped to the Swift package, CLI, docs, or public repo automation.
- Install SwiftLint and ripgrep (`brew install swiftlint ripgrep`) and run `./scripts/check.sh` before opening a PR.
- Add or update tests when changing command parsing, model resolution, inference behavior, or security-sensitive defaults.
- Update `README.md`, `docs/`, and `CHANGELOG.md` when you change public setup, behavior, or operator-facing flags.
- Run `./scripts/update-docs-command-reference.sh` after adding, removing, renaming, or redescribing a CLI command; CI rejects command inventories, docs ownership, navigation, and examples that drift from the code.
- If you touch vendored artifacts or third-party package pins, update [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).
- Do not add hosted-service, billing, or app-store-only surfaces back into this repo.

## Continuous integration lanes

`./scripts/check.sh` is the local gate for every change. It stays the one command
you run before opening a pull request, whatever you touched.

Continuous integration scopes itself to the paths a pull request changes so a
change confined to the macOS app does not pay for the whole package test suite.
The `changes` job sorts every changed path into a category and prints the result
in its job summary, together with the lane it selected:

| Lane | Selected when | What the `swift` job runs |
| --- | --- | --- |
| Fast | Only `apps/macos/**`, `apps/ios/**`, `docs/**`, Markdown, ordinary `scripts/**`, or `assets/`, `integrations/`, `skills/` changed | SwiftLint, the agent readiness, evaluation boundary, and documentation example checks, `swift build`, the MLX metallib verification, and `MereRunAppTests` |
| Full | Anything under `Sources/**`, `Tests/**`, `vendor/**`, `Package.swift`, `Package.resolved`, `.github/**`, or the gate scripts themselves changed | `./scripts/check.sh`, which is every package test target |

Both lanes build and verify the ad-hoc `MereRun.app` bundle, and both verify the
vendored MLX Metal library against the pinned kernel sources. Path classification
is default-deny: a path no rule recognises selects the full lane everywhere, so
a new top-level directory can never take the short route by accident.

Two more scopes come from the same classification. The Linux CLI gate runs when
a change can reach the Linux CLI or the scripts it invokes, and reports success
from a no-op step otherwise, so its required check always reports. The iOS job
runs only when a change can reach the iOS project.

## Merge queue

`main` merges through a merge queue. GitHub builds a `merge_group` ref for each
queued pull request and waits for the six required checks — `swift`,
`macos-app-bundle`, `linux-cli`, `linux-cli-compatibility-docs`,
`dependency-review`, and `secret-scan` — to report on it, so both `ci.yml` and
`security.yml` also trigger on `merge_group`.

Two rules keep the queue moving:

- Every required check runs and reports under its own name on a merge group.
  A job with nothing to prove reports success from a no-op step rather than
  being skipped, because a required check that never reports leaves a merge
  group waiting until it times out. `dependency-review` is the clearest case:
  the action it wraps only supports pull requests, so on a merge group the job
  reports that the pull request's own review already ran.
- Lane selection still applies. A merge group diffs against
  `github.event.merge_group.base_sha`. If that diff cannot be computed the run
  falls back to the full gate, because a merge group is the last check before
  `main`.

Merge-queue runs are never cancelled by concurrency, since a cancelled run
reports nothing and stalls the queue.

The macOS job restores the SwiftPM build directory from a cache that only pushes
to `main` write. Its key covers the Xcode version, `Package.swift`,
`Package.resolved`, and the lane. The MLX Metal library is never restored or
saved: a stale one silently corrupts inference, so every run regenerates and
re-stamps it from the checkout.

## Writing documentation

Follow the [documentation style guide](./docs/documentation-style.md) for public
pages, command descriptions, and examples. In particular:

- write directly to the reader in active voice and present tense
- use sentence case for titles and headings
- use descriptive link text and accessible document structure
- use reserved domains and fictional identifiers in examples
- identify dated evidence and historical benchmark receipts precisely

Run the focused documentation checks before you open a pull request:

```bash
bash ./scripts/check-docs-examples.sh
pnpm docs:build
```

## Opening issues

- Use GitHub issues for bugs, docs gaps, and feature requests.
- Do not post security vulnerabilities in public issues. Follow [`SECURITY.md`](./SECURITY.md) instead.
- Include your macOS version, Xcode/Swift toolchain, the command you ran, and any focused reproduction steps.

## Pull requests

- Keep PRs reviewable and focused on one change set.
- Call out user-visible behavior changes, new flags, vendored artifact updates, or docs changes in the PR description.
- If you intentionally skip a test or validation step, explain why.

By submitting a contribution, you agree that it may be distributed under the MIT license for this project.
