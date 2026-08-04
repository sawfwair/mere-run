# Agent Status

## Purpose

Inspect machine, Pi, provider, and setup-model readiness without changing the
local installation.

## Required Models

No model is required. Installed state is reported for every startable setup
agent that this machine can use.

## Install And Check

```bash
mere.run agent status
mere.run agent status --json
mere.run guide agent status
```

## Parameters

- `--pi-path`: inspect a specific Pi executable instead of automatic discovery.
- `--json`: emit the typed readiness snapshot used by Studio and automation.

## Usage Patterns

- Check whether Pi is installed and whether it is the managed mere.run install.
- Confirm the generated `mere-run` provider extension, host, port, and model.
- Select the recommended installed model before `agent start`.
- Use `--json` for a client that needs stable machine, provider, and model
  readiness fields.
- Open **Serving & Agents → Agents & Clients** in the macOS Studio for the same
  contract with install, configure, start, and connection actions.

## Examples

```bash
mere.run agent status
mere.run agent status --json
mere.run agent status --pi-path "$HOME/bin/pi" --json
```

## Iteration Tips

- Re-run after installing Pi, changing the provider endpoint, or pulling an
  agent model.
- `recommendedModelID` is the best startable tier for this machine; it does not
  imply the model is already installed.
- `provider.configured` requires both the provider record and generated Pi
  extension to exist.

## Troubleshooting

- Pi not found: run `mere.run agent install-pi` or pass `--pi-path`.
- Provider not configured: run
  `mere.run agent onboard --configure-pi --model <id>`.
- Recommended model not installed: run `mere.run model pull <id>` and
  review and accept model terms when required.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/AgentCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Support/PiAgentIntegration.swift
