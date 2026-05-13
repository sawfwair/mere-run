# Setup

## Purpose

Choose a guided, bring-your-own-agent, or manual setup path for a new mere.run installation.

## Required Models

No model is required to view the plan. Agent setup selects this Mac's supported tier; on 96 GB+ Apple Silicon Macs that is `text-agent-deepseek-v4-flash`. Qwen/Q35 agent models are lower-memory or comparison alternatives, not upgrades from DeepSeek V4 Flash.

## Install And Check

```bash
mere.run setup
mere.run setup --mode manual --dry-run
mere.run model capabilities --recommended
```

## Parameters

- `--mode`: `agent`, `byoa`, or `manual`.
- `--agent-model`: `small`, `tier`, or `premier`.
- `--install`: install selected dependencies and models.
- `--start`: start the selected setup path after installing.
- `--dry-run`: print the planned steps.
- `--host`: local API host for the Pi-backed setup agent.
- `--port`: local API port.
- `--pi-path`: explicit Pi executable path.
- `--quiet`, `-q`: suppress install progress.

## Usage Patterns

- Use interactive `mere.run setup` for humans.
- Use `--mode manual --dry-run` for docs or scripts.
- Use agent mode only on Apple Silicon macOS with a supported model. Prefer `--agent-model tier` unless the user asks for a smaller comparison model.

## Examples

```bash
mere.run setup --mode manual --dry-run
```

```bash
mere.run setup --mode agent --agent-model small --install --start
```

## Iteration Tips

- Run capabilities first when the user is unsure what their Mac can run.
- Start with the small agent model on lower-memory machines; keep DeepSeek V4 Flash as the preferred tier/premier agent on 96 GB+ Macs.
- Use BYOA when the user already prefers Claude, Codex, or another local tool.

## Troubleshooting

- No supported agent model: use `--mode byoa` or `--mode manual`.
- Pi not found: run `mere.run agent install-pi`.
- Start fails because model is missing: pull the selected model or rerun with `--install`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/SetupCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCore/AgentModelResources.swift
