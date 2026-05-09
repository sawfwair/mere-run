# Model Capabilities

## Purpose

Report which managed models this Mac can run, including memory fit, recommendations, categories, and reasons unsupported models are hidden or rejected.

## Required Models

No model is required.

## Install And Check

```bash
mere.run model capabilities
mere.run model capabilities --recommended
mere.run model capabilities --all
```

## Parameters

- `--all`: include unsupported models and reasons.
- `--recommended`: show only supported first-setup recommendations with managed download sources.

## Usage Patterns

- Run this before large downloads.
- Use `--recommended` for first-time setup.
- Use `--all` when a pull is blocked by hardware support.

## Examples

```bash
mere.run model capabilities --recommended
```

```bash
mere.run model capabilities --all
```

## Iteration Tips

- Choose the smallest supported model that satisfies the workflow.
- Prefer recommended models for first-run UX.
- When a supported recommendation needs a local path, explain that it is not pullable in the public build.

## Troubleshooting

- No recommendations: the machine may be unsupported or too memory-constrained.
- A desired model is hidden: rerun with `--all`.
- A pull is blocked: use a supported model or consciously pass `--allow-unsupported` to `model pull`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/ModelCapabilitiesCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCore/ManagedModelCapabilityCatalog.swift
