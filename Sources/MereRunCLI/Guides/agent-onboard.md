# Agent Onboard

## Purpose

Summarize this Mac's model capabilities and optionally prepare the Pi coding-agent integration for guided local setup.

## Required Models

No model is required to print readiness. Optional model pulls/configuration
should use the recommended setup-agent tier; on 96 GB+ Apple Silicon Macs that
is `text-agent-deepseek-v4-flash`. Tool-capable native chat models such as
`text-agent-ornith-9b` are lower-memory alternatives. GGUF models served by
the `text-code` engine remain useful for direct coding experiments and evals,
but that API lane rejects tool calls and is not exposed to Pi.

## Install And Check

```bash
mere.run agent onboard
mere.run agent onboard --help
```

## Parameters

- `--pull-recommended`: pull supported first-setup model packages.
- `--accept-model-license`: confirm that you reviewed and accept the listed third-party terms and agree to comply with them before restricted recommended-model downloads.
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
- For Ornith 35B MLX, install `text-agent-ornith-35b-mlx`, start `api serve --engine text-chat-q36 --model text-agent-ornith-35b-mlx`,
  then use `--configure-pi --model text-agent-ornith-35b-mlx --host <host> --port <port>`.
- For Ornith, pull `text-agent-ornith-9b`, start `api serve --engine text-chat-q36 --model text-agent-ornith-9b`,
  then use `--configure-pi --model text-agent-ornith-9b --host <host> --port <port>`.

## Examples

```bash
mere.run agent onboard
```

```bash
mere.run agent onboard --install-pi --configure-pi --model text-agent-deepseek-v4-flash
```

```bash
mere.run model pull text-agent-ornith-9b
mere.run agent onboard --configure-pi --model text-agent-ornith-9b --port 8080
```

## Iteration Tips

- Copy the printed recommended setup-agent id into later `model pull` or `agent start` commands.
- Keep host and port aligned with `api serve`.
- Use `mere.run status --host <host> --port <port>` after starting a server.
- Re-run after changing hardware, model store, or setup model choice.

## Troubleshooting

- Provider model unsupported: choose a tool-capable model printed as startable
  by onboarding; `text-code` models cannot run Pi tools.
- Pi install fails: rerun with network access and without `--quiet`.
- No recommended downloads: use manual setup or a local model path.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/AgentCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCore/AgentModelResources.swift
