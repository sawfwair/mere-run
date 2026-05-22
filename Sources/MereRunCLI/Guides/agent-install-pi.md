# Agent Install Pi

## Purpose

Install the latest Pi coding-agent release so mere.run can launch a guided local setup agent. Auto-install uses the published macOS release assets; on Linux, install Pi separately and pass `--pi-path` or put `pi` on PATH.

## Required Models

No model is required.

## Install And Check

```bash
mere.run agent install-pi
mere.run agent install-pi --help
```

## Parameters

- `--force`: re-download and replace the current installed Pi release.

## Usage Patterns

- Run before `agent start` on macOS when Pi is not on PATH or not managed by mere.run.
- Use `--force` when the installed Pi binary is corrupt or outdated.
- Pair with `agent onboard --configure-pi` to register the local provider.

## Examples

```bash
mere.run agent install-pi
```

```bash
mere.run agent install-pi --force
```

## Iteration Tips

- Keep the printed Pi path for debugging.
- Reinstall after a failed partial download.
- Use `agent onboard` after install to configure provider metadata.

## Troubleshooting

- Network failure: retry when GitHub release downloads are reachable.
- Still not found: pass `--pi-path` to `agent start`.
- Linux: provide an existing Pi binary with `--pi-path` or PATH.
- Wrong version: use `--force`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/AgentCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Support/PiAgentIntegration.swift
