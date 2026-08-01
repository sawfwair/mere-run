# Model Management

One store, canonical IDs, and honest accounting. `mere.run` checks whether this
machine can run a model before it spends your bandwidth on it, reports what a
model actually costs on disk once shared payloads are counted once, and shows
exactly what is safe to delete.

## Commands

| Command | What it does |
| --- | --- |
| `mere.run model capabilities` | Show which managed models this machine can run. |
| `mere.run model list` | List all known models with install status. |
| `mere.run model pull` | Download a managed model into the local model store. |
| `mere.run model info` | Print a model's manifest, validation status, and resolved component paths. |
| `mere.run model remove` | Remove a model from the local model store. |
| `mere.run model storage` | Inspect physical model storage, sharing, and reclaimable space. |
| `mere.run model gc` | Find or delete unreferenced model payloads and partial downloads. |
| `mere.run model repair-manifests` | Rewrite missing manifest metadata in the local store. |
| `mere.run model runtime` | Read and update per-model API runtime settings. |
| `mere.run model benchmark` | Run focused local model benchmarks. |
| `mere.run adapter list` | List cataloged LoRA adapters and their install state. |
| `mere.run adapter pull` | Download and verify a cataloged LoRA adapter. |
| `mere.run status` | Show local server, loaded model, and installed model status. |
| `mere.run setup` | Choose a guided, BYOA, or manual setup path. |

## Default model store

By default:

```text
~/Library/Application Support/MereRun/models
```

Override with:

```bash
export MERERUN_MODELS_DIR=/path/to/models
```

or:

```bash
swift run mere.run --models-root /path/to/models model list
```

## macOS Studio

Open **Models**, switch the scope to **All**, and select any missing model to
download it directly. Studio streams the underlying `model pull` output, keeps
cancelled partial downloads resumable, and refreshes the inventory after a
successful install. Models with restricted third-party terms show their source
links and require **Acknowledge & Download** before transfer. The **Files**
button opens the configured model store in Finder; it does not start a download.

## Canonical model IDs

Examples:

- images: `image-klein-nano`, `image-bonsai-binary`, `image-bonsai-ternary`, `image-zimage-nano`, `image-klein-max`, `image-zimage-max`
- text: `text-chat-gemma4`, `text-chat-laguna-s-2-1`, `text-chat-laguna-xs-2-1`, `text-chat-q36-nano`, `text-chat-bonsai-27b-1bit`, `text-chat-bonsai-27b-2bit`, `text-chat-lfm25-a1b-8bit`, `text-agent-deepseek-v4-flash`, `text-agent-qwen35-9b`, `text-agent-ornith-9b`, `text-agent-ornith-35b-mlx`, `text-agent-ornith-35b`, `text-code-north-mini`, `text-code-qwen3`, `text-embed-qwen3-0.6b`
- speech: `speech-tts-qwen3-nano`, `speech-asr-parakeet`
- vision: `vision-ocr-lighton`
- music: `music-acestep`, `music-acestep-xl-turbo`, `music-acestep-xl-turbo-lm4b`, `music-acestep-xl-sft`, `music-acestep-xl-base`, `music-magenta-rt2-small`, `music-magenta-rt2-base`
- sfx: `sfx-woosh-dflow`, `sfx-woosh-flow`, `sfx-woosh-clap`, `sfx-woosh-synchformer`, `sfx-woosh-dvflow-8s`, `sfx-woosh-vflow-8s`, `sfx-mmaudio-large-44k-v2`
- video: `video-ltx-av`, `video-ltx23-av-mlx`, `video-ltx23-full-mlx`, `video-ltx23-a2vid-mlx`, `video-wan22-ti2v-5b-mlx`, `video-scail2-14b-mlx`

The public runtime resolves these IDs directly, so docs and examples should use
the canonical names shown by `mere.run model list`.

## Runtime entrypoints

- `Sources/MereRunCore/MereRunModelPaths.swift`
- `Sources/MereRunCore/MereRunModelManifest.swift`
- `Sources/MereRunCore/ModelResolver.swift`

## Command responsibilities

### `mere.run model list`

Shows the canonical managed model table, installed status, and referenced size.
Referenced sizes are not additive because multiple models can share payload files.

### `mere.run model storage`

Reports physical application-support, Hub, model-store, and other bytes, plus
safe-to-collect partial/orphaned data. Per-model rows distinguish referenced,
reclaimable-on-removal, and shared bytes. Pass `--json` for byte-exact output.

### `mere.run model gc`

Builds a read-only cleanup plan by default. `--force` recomputes and validates
that plan under the cross-process storage lock before deleting unreferenced
cache units, stale payloads, partial downloads, dead revision references, and
unlinked blobs. Newly created unreferenced snapshots have a one-hour grace
period. Installed model links and legacy links under MereRun application support
keep their backing payloads live.

### `mere.run status`

Combines the model inventory with a local API probe. It reports the active
model-store path/source, installed managed models, and whether the configured
API server answers `/health`. When the server exposes `/runtime/status`, that
snapshot (active models, admission, batching, KV reuse, cache/memory stats) is
the primary readout; the model IDs from `/v1/models` are the fallback.

### `mere.run model capabilities`

Summarizes the current machine, the managed models it can run, the preferred
setup-agent tier, chat winners by RAM band, and cross-modality starter coverage.
Pass `--all` to include models that are blocked by platform or memory
requirements.

### `mere.run model info`

Shows the resolved local install for one canonical model ID.

### `mere.run model pull`

Downloads a managed model from its cataloged Hugging Face source. Pulls are
checked against the managed capability catalog before download so low-memory
machines do not fetch models they cannot run. Pass `--allow-unsupported` only
when you intentionally accept that risk or are using external hardware.

Access-gated models and models with material non-commercial, research-only, or
revenue-limited terms require `--accept-model-license` for new downloads and
never auto-download at runtime. A custom license alone does not trigger the
flag. The CLI and macOS app show the
applicable model/component terms before acceptance; schema 3 of the installed
`mererun_model.json` records the immutable source revisions, term URLs, and
acknowledgement. mere.run does not determine whether a user's intended use is
permitted. The complete inventory is in
[`model-sources.md`](../model-sources.md#restricted-model-downloads).

Laguna XS 2.1 is released under the permissive OpenMDW-1.1 license, and its
public Hugging Face repository is not gated. It therefore installs without a
separate acknowledgement flag while retaining the upstream license file:

```bash
mere.run model pull text-chat-laguna-xs-2-1
mere.run text chat --model text-chat-laguna-xs-2-1 --prompt "Hello"
```

### `mere.run adapter list` and `mere.run adapter pull`

Adapters use a separate checksum-pinned catalog and install under:

```text
~/Library/Application Support/MereRun/adapters/<adapter-id>/<version>/
```

Catalog entries include the promoted Mere Platform Assistant v22 and the
remote-only Apache-2.0 LightX2V Wan 2.1 I2V four-step adapter:

```bash
mere.run model pull text-chat-gemma4-12b-4bit
mere.run adapter list
mere.run adapter pull mere-platform-assistant
mere.run adapter pull scail2-lightx2v-4step
mere.run text chat \
  --model text-chat-gemma4-12b-4bit \
  --lora mere-platform-assistant \
  --prompt "What can you help me with in Mere?"
```

`adapter pull` downloads the immutable upstream artifact, verifies its exact
byte count and SHA-256, and prints the installed adapter path on stdout.
Catalog ids are accepted by `text chat --lora`, `api serve --lora`, and the
compatible SCAIL `video animate --distilled-adapter` surface; local paths
remain supported. The default `video animate --profile fast` selects the SCAIL
adapter ID and its published four-step schedule automatically.

### `mere.run setup`

Guided onboarding for the shared model store and first local agent. The command
offers a Pi-powered Mere agent, a BYOA prompt for Claude/Codex, or manual
commands. The small local agent model is `text-agent-qwen35-9b`; hardware-tier
setup can select Qwen3.6 nano, Qwen3-Coder Next, or DeepSeek V4 Flash. On 96 GB+
Apple Silicon Macs, `text-agent-deepseek-v4-flash` is the preferred managed
setup-agent tier; smaller Qwen agent models are alternatives, not upgrades.
`text-code-north-mini` can be pulled, inspected, and run through the native
GGUF code runtime for coding-agent comparisons against `text-code-qwen3`.
`text-agent-ornith-9b` can be pulled, inspected, and run through the native
Qwen-family MLX/OptiQ runtime for coding-agent comparisons.
`text-agent-ornith-35b-mlx` is a local-only converted MLX Q4 Ornith target; it
can be inspected and served once its converted directory exists in the model
store, but `model pull` is disabled until a converted public snapshot exists.
`text-agent-ornith-35b` is the larger GGUF Ornith eval target and runs through
the native `text-code`/llama.cpp path.

### `mere.run model remove`

Removes a managed install and its unshared backing payloads, while preserving
files referenced by another managed or legacy consumer. Confirmation and output
show referenced versus reclaimable bytes. Pass `--keep-cache` to remove only the
install links, or `--force --json` for structured automation.

### `mere.run model repair-manifests`

Repairs missing manifest metadata for known models in the local store. Preview
the exact writes and consume a typed report with:

```bash
mere.run model repair-manifests --dry-run --json
mere.run model repair-manifests --json
```

The report identifies healthy manifests, proposed or completed writes, absent
model directories, and write errors without treating models that were never
installed as damaged.

In macOS Studio, open **Models → Health**. The workspace presents the structured
manifest audit, confirms repair before writing, and runs `mere.run gate` suites
as durable Library jobs. Correctness suites can be selected independently,
strict performance thresholds are opt-in, and baseline replacement requires a
separate warning confirmation. Every gate writes a JSON report.

### `mere.run model runtime`

Reads and updates typed per-model API runtime settings — how a model behaves
when served by `mere.run api serve`. `get` prints the stored settings for one
managed model id or configured alias; `set` updates them:

The macOS Studio's top-level **Serving & Agents → Model Pool** view uses these
same settings for both text models and managed sidecars. Text rows additionally
support explicit HTTP load/unload. Sidecars load on first request and expose
their lifecycle, readiness, queue, replacement, failure, and eviction state;
their managed pin/TTL controls continue to go through `model runtime set`.

```bash
mere.run model runtime get text-chat-gemma4
mere.run model runtime set text-chat-gemma4 \
  --alias gemma --pinned --ttl-seconds 600 \
  --max-context-tokens 8192 --temperature 0.7
```

`set` accepts `--alias` (request-facing alias), `--pinned`/`--unpinned` (keep
the model loaded across automatic TTL/LRU eviction), `--ttl-seconds` (unload
TTL), `--max-context-tokens`, `--max-tokens`, `--temperature`, `--top-p`,
`--min-p`, `--engine` (engine override, validated against catalog
compatibility), and
`--kv-cache-mode` (`default`, `affine4`, `affine8` — `affine8` applies to
Gemma4/Qwen/LFM2 — plus Gemma4-only `polar2`/`auto`). Each option has a
matching clear flag (`--clear-alias`, `--clear-ttl`,
`--clear-max-context-tokens`, `--clear-max-tokens`,
`--clear-temperature`, `--clear-top-p`, `--clear-min-p`, `--clear-engine`,
`--clear-kv-cache-mode`) to remove the stored value. Both subcommands accept
`--json`. Only models with configurable API residency are accepted.

### `mere.run model benchmark`

Runs focused local model benchmarks. This is a developer and performance tool,
not part of the everyday inference workflow. Lanes:

- `chat` — runs a small grounded-chat eval slice against local assistant models.
- `tool-calls` — runs a small tool-call selection eval against local chat models.
- `tool-continuations` — evaluates Gemma 4 continuation after completed tool
  calls.
- `code` — runs a real coding-eval slice against local coding models.
- `gemma4-kv` — compares Gemma4 default KV cache decode against packed PolarKV.
- `gemma4-mtp` — compares Gemma4 serial decode against verified MTP speculative
  decode.
- `q36-mtp` — compares Qwen3.6 serial decode against adaptive and forced MTP
  speculative decode.
- `api-workload` — replays a chat workload against a running API server and
  measures runtime cache counters.
- `vlm` — compares vision-language chat models on synthetic or lmms-eval
  datasets.

Each lane accepts `--json` for machine-readable reports; see
`mere.run model benchmark <lane> --help` for lane-specific flags.

## Related docs

- [Model Sources](../model-sources.md)
- [Configuration](../configuration.md)
- [Testing Guide](../testing.md)
