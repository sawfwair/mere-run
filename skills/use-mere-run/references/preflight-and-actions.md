# Preflight and actions

Use preflight before a download or expensive run when that command supports
it. Discover support from the selected command's `--help`; do not append the
same flags to every modality.

## Choose the preparation mode

| Surface | How to prepare |
| --- | --- |
| `model pull` | `--preflight --json` inspects support, terms, sources, disk, and installation without downloading. |
| Image/video generation, image training, supported vision/geo operations, API serving | Use their advertised `--preflight --json`; checks and schema differ by command. |
| `text chat` | `--preflight --json`; add `--require-installed` to both check and run when downloads must be explicit. |
| Workflow graphs | `graph validate` checks the document; `graph preflight` also probes the selected executor. |
| Reconstruction, geometry, text training, evaluation, benchmarks | Some use `--dry-run`. Read that command's semantics; dry runs can prepare files and are not interchangeable with preflight. |
| Speech synthesis/transcription, music generation, other commands without preflight | Preflight the model pull, inspect inputs, read help and the handbook, and select a bounded first run. Do not invent `--preflight`. |

`--json` can be restricted to preflight. During generation, a result receipt is
a different output contract. Never combine `--receipt` with `--preflight`.

## Read the report, including failed checks

Capture stdout and stderr separately and retain the exit code. A blocked check
can print valid JSON and then exit nonzero; do not discard its stdout because a
shell command failed. Syntax errors can instead produce only stderr.

For the shared envelope, inspect:

- `schema_version`, `command`, and `mode` to identify the contract.
- `status`: `blocked` prevents execution; `warning` requires reading the warning,
  and `ok` means only that the implemented checks passed.
- `diagnostics`: read `severity`, `id`, `message`, locations, and
  `suggested_action_ids`. Resolve blockers; use warnings and estimates to adjust
  the plan when they affect the user's requirements.
- `request` and `result`: confirm model, inputs, output path, dimensions, seed,
  resolved settings, download behavior, and execution host.
- `actions`: suggested follow-up operations, not permission to execute them.

Do not assume every JSON command has this envelope. In particular, text-chat
preflight returns a compact report with `schema_version`, `status`, `model`,
`installed`, optional `model_path`, and `diagnostics`. It can return **exit 0
with `status: "blocked"`**. Without `--require-installed`, an uninstalled chat
model can have `status: "ok"`. Check both installation and report status.

Model-pull reports include `result.models` entries with `supported`,
`installed`, `runtime_ready`, `conversion_required`, `has_download_source`,
`will_download`, `companion_model_ids`, and `usage_terms`. A download-ready
report does not mean the model is ready for inference. Inspect both the model
store and Hub cache paths and headroom. Byte estimates can be absent or differ
from eventual use; missing estimates do not mean zero bytes.

## Work an image request through preparation

Adapt the model to the machine and the prompt to the user's request. First
inspect the download; this command does not pull the model:

```bash
mere.run model pull image-zimage-nano --preflight --json
```

After resolving the reported blockers, download the selected model and check
the exact generation request:

```bash
mere.run model pull image-zimage-nano
mere.run image generate --model image-zimage-nano \
    --prompt "a ceramic mug in soft morning light" --seed 42 \
    --output ./mug.png --preflight --json
```

Read the report before proceeding. If it permits the run and the output path
is suitable, keep the same generation arguments and change only output modes:

```bash
mere.run image generate --model image-zimage-nano \
    --prompt "a ceramic mug in soft morning light" --seed 42 \
    --output ./mug.png --receipt --progress-json
```

Do not carry preflight's `--json` into a command that accepts it only with
`--preflight`. Preserve a global model-store override, working directory, cache
settings, environment, and selected executable across all three stages.

For chat, explicitly prevent a hidden download after installation:

```bash
mere.run model pull text-chat-gemma4-12b-4bit --preflight --json
```

After the pull preflight passes:

```bash
mere.run model pull text-chat-gemma4-12b-4bit
mere.run text chat --model text-chat-gemma4-12b-4bit --require-installed \
    --prompt "Explain local inference in one paragraph." --preflight --json
```

After reading the chat report and resolving blockers:

```bash
mere.run text chat --model text-chat-gemma4-12b-4bit --require-installed \
    --prompt "Explain local inference in one paragraph." --stream
```

## Apply follow-up actions deliberately

An action can include `enabled`, `disabled_reason`, `requires`, `confirmation`,
`destructive`, `secrets_masked`, and a `command` containing `argv` and `cwd`.
Read these fields before using it. Choose an enabled action relevant to the
user's task and existing authorization; do not run every offered action.

Treat `command.argv` as literal arguments to a process, not shell code. Do not
join it into a string and evaluate it. Preserve `cwd`, the selected executable,
global overrides, and required environment. A masked action can omit secrets;
reconstruct required credentials from the authorized environment rather than
running placeholder values or exposing them in logs.

A suggested pull on an SSH or relay worker must target that worker's model
store. Running it locally does not repair the remote preflight. A suggested
`--allow-unsupported`, license-acceptance flag, deletion, or force option is not
automatic authorization. Resolve the concrete requirement, then preflight the
original request again. If the same blocker repeats unchanged, report it and
the exact dependency instead of retrying indefinitely.
