# Agent Start

## Purpose

Start Pi against a local mere.run setup-agent API server. This is the guided "help me set up mere.run" runtime path.

## Required Models

Use this machine's supported setup-agent tier. On 96 GB+ Apple Silicon Macs, `text-agent-deepseek-v4-flash` is the preferred managed setup agent. Smaller Qwen, North Mini Code, and Ornith 35B models are lower-memory or comparison alternatives, not upgrades from DeepSeek V4 Flash. On Linux, provide Pi with `--pi-path` or PATH; auto-install uses macOS release assets.

## Install And Check

```bash
mere.run agent onboard
mere.run model pull text-agent-deepseek-v4-flash
mere.run agent install-pi
mere.run agent start --model text-agent-deepseek-v4-flash
mere.run status
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

- Run `model capabilities --recommended` or `agent onboard` first and use the recommended setup-agent id.
- Pull the selected model before `agent start`.
- Use `--skip-server` only when you already started a compatible local API server.
- Use `agent start --model text-code-north-mini` to compare North Mini Code
  through the native GGUF code runtime.
- Use `agent start --model text-agent-ornith-35b` to compare the larger Ornith
  GGUF coding-agent target through the same native `text-code` runtime.
- On Linux, provide an existing Pi binary with `--pi-path` or PATH before starting.
- Run `status` when you need to confirm the local server and served model.
- Keep the default prompt unless the user has a specific setup goal.

## Examples

```bash
mere.run agent start --model text-agent-deepseek-v4-flash
```

```bash
mere.run agent start \
  --model text-agent-deepseek-v4-flash \
  --prompt "Help me install only speech and OCR models."
```

## Iteration Tips

- Use DeepSeek V4 Flash on 96 GB+ Macs; start with a smaller Qwen agent only on lower-memory machines or when comparing behavior.
- Check the server log path printed to stderr when startup hangs.
- Re-run onboarding after model pulls or provider changes.

## Troubleshooting

- Model unsupported: choose a supported model from `agent onboard`.
- Model missing: run `mere.run model pull <id>`.
- Pi missing: run `mere.run agent install-pi` on macOS, or pass `--pi-path` / put `pi` on PATH on Linux.
- Health check times out: verify host/port and local API logs.
- Unsure what is already running: run `mere.run status --host <host> --port <port>`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/AgentCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCore/AgentModelResources.swift
