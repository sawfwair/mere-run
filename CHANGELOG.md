# Changelog

All notable changes to this public repository will be documented in this file.

The format is based on Keep a Changelog and uses a simple `Unreleased` section
until tagged public releases begin.

## Unreleased

### Added

- third-party notices for vendored runtime artifacts
- GitHub issue templates, pull request template, Dependabot, and a lightweight security workflow

### Changed

- clarified public contributor guidance and removed maintainer-specific workflow files from the repo surface
- documented local API serving safety defaults, advanced operator flags, and HTTPS-only remote artifact expectations
- hardened API request validation for generation parameters and operator-controlled LoRA selection
- kept `shell_exec` out of non-interactive tool auto-approval even when `--auto-approve-tools` is passed
- made signed model-source endpoint failures fail closed unless fallback is explicitly allowed
- added SHA-256 verification support for managed archive downloads from catalog pins or `MERERUN_MODEL_SOURCE_SHA256S`
- hardened recoverable runtime construction and conditioning failures to throw typed errors instead of terminating the process

### Fixed

- fixed DMG installs when the optional packaged model-source URL sidecar is absent
