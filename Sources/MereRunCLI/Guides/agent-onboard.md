# Agent Onboard

## Purpose

Summarize this Mac's model capabilities and optionally prepare the Pi coding-agent integration for guided local setup.

## Required Models

No model is required to print readiness. Optional model pulls/configuration should use the recommended setup-agent tier; on 96 GB+ Apple Silicon Macs that is `text-agent-deepseek-v4-flash`. Qwen/Q35 agent models are lower-memory or comparison alternatives.

## Install And Check

```bash
mere.run agent onboard
mere.run agent onboard --help
```

## Parameters

- `--pull-recommended`: pull supported first-setup model packages.
- `--install-pi`: install the latest Pi coding-agent release.
- `--configure-pi`: write a Pi extension for the local mere.run API provider.
- `--host`: local API host for the provider extension.
- `--port`: local API port.
- `--model`: model id to expose in the provider extension.
- `--quiet`, `-q`: suppress install progress.

## Usage Patterns

- Run plain `agent onboard` first; it is informational.
- Use `--install-pi` before `agent start` if Pi is not already installed.
- Use `--configure-pi --model <id>` when Pi should call a local mere.run API provider.

## Examples

```bash
mere.run agent onboard
```

```bash
mere.run agent onboard --install-pi --configure-pi --model text-agent-deepseek-v4-flash
```

## Iteration Tips

- Copy the printed recommended setup-agent id into later `model pull` or `agent start` commands.
- Keep host and port aligned with `api serve`.
- Re-run after changing hardware, model store, or setup model choice.

## Troubleshooting

- Provider model unsupported: choose a model printed by onboarding.
- Pi install fails: rerun with network access and without `--quiet`.
- No recommended downloads: use manual setup or a local model path.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/AgentCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCore/AgentModelResources.swift
