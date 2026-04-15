# Contributor Notes

This repo is public and contributor-facing. Keep the guidance here generic, project-specific, and useful to someone working from a clean checkout.

## Working agreement

- keep changes scoped to the Swift package, CLI, docs, and public tooling in this repo
- run `./scripts/check.sh` before opening a pull request, plus any focused tests for the area you changed
- update `README.md`, `docs/`, or `CHANGELOG.md` when you change public CLI behavior, setup, or security-sensitive defaults
- do not reintroduce hosted-service, billing, or private deployment assumptions into this repository
- if you touch vendored artifacts or package pins, update [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) with provenance and license details

## Public-repo boundaries

- prefer documented, repo-local workflows over maintainer-specific editor automation
- do not commit machine-local configuration, secrets, or environment-specific shortcuts
- keep contributor guidance neutral and tool-agnostic unless the tool is part of the public repo experience
