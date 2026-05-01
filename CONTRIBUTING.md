# Contributing

- Keep changes scoped to the Swift package, CLI, docs, or public repo automation.
- Install SwiftLint and ripgrep (`brew install swiftlint ripgrep`) and run `./scripts/check.sh` before opening a PR.
- Add or update tests when changing command parsing, model resolution, inference behavior, or security-sensitive defaults.
- Update `README.md`, `docs/`, and `CHANGELOG.md` when you change public setup, behavior, or operator-facing flags.
- If you touch vendored artifacts or third-party package pins, update [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).
- Do not add hosted-service, billing, or app-store-only surfaces back into this repo.

## Opening issues

- Use GitHub issues for bugs, docs gaps, and feature requests.
- Do not post security vulnerabilities in public issues. Follow [`SECURITY.md`](./SECURITY.md) instead.
- Include your macOS version, Xcode/Swift toolchain, the command you ran, and any focused reproduction steps.

## Pull requests

- Keep PRs reviewable and focused on one change set.
- Call out user-visible behavior changes, new flags, vendored artifact updates, or docs changes in the PR description.
- If you intentionally skip a test or validation step, explain why.

By submitting a contribution, you agree that it may be distributed under the MIT license for this project.
