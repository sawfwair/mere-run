# Agent Start

## Purpose

Start Pi against a local mere.run setup-agent API server. This is the guided "help me set up mere.run" runtime path.

## Required Models

Use a supported managed agent model such as `text-code-qwen3` or `text-agent-qwen35-9b`, depending on this Mac's capability report.

## Install And Check

```bash
mere.run agent onboard
mere.run model pull text-code-qwen3
mere.run agent install-pi
mere.run agent start --model text-code-qwen3
```

## Parameters

- `--host`: local API host.
- `--port`: local API port.
- `--pi-path`: explicit Pi executable path.
- `--prompt`: initial prompt sent to Pi.
- `--model`: managed agent model id.
- `--skip-server`: use an already-running mere.run API server.
- `--allow-unsupported`: start even if hardware support marks the model unsupported.

## Usage Patterns

- Run `agent onboard` first and use one of its printed model ids.
- Pull the selected model before `agent start`.
- Use `--skip-server` only when you already started a compatible local API server.
- Keep the default prompt unless the user has a specific setup goal.

## Examples

```bash
mere.run agent start --model text-code-qwen3
```

```bash
mere.run agent start \
  --model text-agent-qwen35-9b \
  --prompt "Help me install only speech and OCR models."
```

## Iteration Tips

- Start with the smallest supported model for setup guidance.
- Check the server log path printed to stderr when startup hangs.
- Re-run onboarding after model pulls or provider changes.

## Troubleshooting

- Model unsupported: choose a supported model from `agent onboard`.
- Model missing: run `mere.run model pull <id>`.
- Pi missing: run `mere.run agent install-pi` or pass `--pi-path`.
- Health check times out: verify host/port and local API logs.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/AgentCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCore/AgentModelResources.swift
