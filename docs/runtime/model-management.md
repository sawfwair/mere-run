# Model management

Use model-management commands to work with one writable store, a unified
catalog, canonical IDs, and physical-storage accounting. Before downloading a
model, `mere.run` checks whether the machine can run it. Storage reports count
shared payloads once and identify data that is safe to delete.

## Commands

| Command | What it does |
| --- | --- |
| `mere.run model capabilities` | Show which managed models this machine can run. |
| `mere.run model list` | List all known models with install status. |
| `mere.run model location` | Register read-only catalog roots and explicit model-directory bindings. |
| `mere.run model pull` | Download a managed model into the local model store. |
| `mere.run model info` | Print a model's manifest, validation status, and resolved component paths. |
| `mere.run model remove` | Remove a model from the local model store. |
| `mere.run model storage` | Inspect physical model storage, sharing, and reclaimable space. |
| `mere.run model gc` | Find or delete unreferenced model payloads and partial downloads. |
| `mere.run model optimize` | Build inference-only caches for a supported installed model. |
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

Explicit `MERERUN_MODELS_DIR` and `--models-root` overrides intentionally use
only that root. This keeps scripts, tests, and remote jobs reproducible. The
persisted primary store selected by the macOS app participates in the unified
catalog normally.

## Registered catalog locations

The primary store is the only writable location. Register existing model
collections on other disks without copying files or building a symbolic-link
facade:

```bash
# A canonical catalog root: /Volumes/Models/<model-id>/
mere.run model location add /Volumes/Models

# An arbitrary existing directory name
mere.run model location bind video-ltx23-full-mlx /Volumes/SALVATION/models/LTX-2.3 \
  --accept-model-license

mere.run model location list
mere.run model location list --json
```

Resolution order is deterministic:

1. the writable primary store
2. explicit bindings, in registration order
3. registered search roots, in registration order

If a removable disk is unavailable, resolution tries the next valid location.
Search-root models must live at `<root>/<canonical-model-id>/` and carry a
matching `mererun_model.json`. A binding supplies the canonical identity for an
arbitrarily named folder; mere.run validates it but does not write a manifest or
any other file into external storage. Registrations live in:

```text
~/Library/Application Support/MereRun/model_locations.json
```

Use `model location remove <root>` or `model location unbind <id> [path]` to
remove registrations. Both operations preserve every payload byte.

## macOS Studio

Open **Models**, switch the scope to **All**, and select any missing model to
download it directly. Studio streams the underlying `model pull` output, keeps
canceled partial downloads resumable, and refreshes the inventory after a
successful install. Models with restricted third-party terms show their source
links and require **Accept & Download** before transfer. Continuing confirms
that you reviewed and accept the listed terms and agree to comply. The
**Files** button opens the configured model store in Finder; it does not start
a download.

## Canonical model IDs

The following list shows representative canonical IDs by modality:

- images: `image-flux1-dev`, `image-flux2-dev`, `image-klein-nano`, `image-bonsai-binary`, `image-bonsai-ternary`, `image-zimage-nano`, `image-klein-max`, `image-zimage-max`
- text, chat, and embeddings: `text-chat-gemma4`, `text-chat-laguna-s-2-1`, `text-chat-laguna-xs-2-1`, `text-chat-nemotron-35-lightning`, `omni-chat-nemotron3-nano-30b-a3b-bf16`, `text-chat-q36-nano`, `vision-chat-q38-27b`, `vision-chat-q38-27b-4bit`, `vision-chat-q38-flash-next-mixed`, `vision-chat-q38-flash-next-3bit`, `vision-chat-q38-flash-next-3bit-native-ple`, `vision-chat-q38-flash-next-4bit`, `text-chat-bonsai-27b-1bit`, `text-chat-bonsai-27b-2bit`, `text-chat-lfm25-1.2b-bf16`, `text-chat-lfm25-1.2b-qad-4bit`, `text-chat-lfm25-2.6b-4bit`, `text-chat-lfm25-2.6b-bf16`, `text-chat-lfm25-2.6b-qad-4bit`, `text-chat-lfm25-a1b-8bit`, `text-chat-lfm25-a1b-bf16`, `vision-chat-lfm25-3b-8bit`, `text-agent-deepseek-v4-flash`, `text-agent-qwen35-9b`, `text-agent-ornith-9b`, `text-agent-ornith-35b-mlx-4bit`, `text-agent-ornith-35b-mlx-6bit`, `text-agent-ornith-35b-mlx-8bit`, `text-agent-ornith-35b-mlx`, `vision-chat-ornith-35b`, `text-agent-ornith-35b`, `text-code-north-mini`, `text-code-qwen3`, `text-embed-qwen3-0.6b`, `vision-embed-qwen3-vl-2b`

Pulling an exact BF16 LFM2.5 DSpark target also pulls its pinned `*-dspark` companion after
the same LFM Open License acknowledgement. Companions remain separately
addressable for repair and inspection but are not standalone chat models.
- speech: `speech-tts-qwen3-nano`, `speech-asr-parakeet`
- vision: `vision-ocr-lighton`
- music: `music-acestep`, `music-acestep-xl-turbo`, `music-acestep-xl-turbo-lm4b`, `music-acestep-xl-sft`, `music-acestep-xl-base`, `music-acestep-lm-1.7b`, `music-acestep-lm-4b`, `music-magenta-rt2-small`, `music-magenta-rt2-base`
- sfx: `sfx-woosh-dflow`, `sfx-woosh-flow`, `sfx-woosh-clap`, `sfx-woosh-synchformer`, `sfx-woosh-dvflow-8s`, `sfx-woosh-vflow-8s`, `sfx-mmaudio-large-44k-v2`
- video: `video-minimax-h3-fl2va-mlx`, `video-minimax-h3-fl2va-bf16-mlx`,
  `video-minimax-h3-ref2va-mlx`, `video-ltx-av`, `video-ltx23-av-mlx`,
  `video-ltx23-full-mlx`, `video-ltx23-a2vid-mlx`,
  `video-ltx25-distilled-bf16`, `video-ltx25-full-bf16`,
  `video-wan22-ti2v-5b-mlx`, `video-scail2-14b-mlx`

The public runtime resolves these IDs directly. Documentation and examples must use
the canonical names shown by `mere.run model list`.

## Runtime entrypoints

- `Sources/MereRunCore/MereRunModelPaths.swift`
- `Sources/MereRunCore/MereRunModelManifest.swift`
- `Sources/MereRunCore/ModelResolver.swift`

## Command responsibilities

### `mere.run model list`

Shows the canonical managed model table and shallow availability without
recursively scanning payload files. Pass `--measure-sizes` when referenced
sizes are needed; those values follow symlinks and are not additive because
multiple models can share payload files.

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

### `mere.run model optimize`

For compatible MiniMax-H3 or LTX 2.5 roots, the command builds the model-specific native
inference artifact:

```bash
mere.run model optimize ./MiniMax-H3-FL2VA-full-MLX
mere.run model optimize video-ltx25-full-bf16
```

The versioned cache pack is written beside the installed model. It contains
exact tables for 5, 9, 12, 16, 21, and 31 points at shifts 12/3 plus the
LightX2V 768p 5-point table at shifts 6/3. Custom schedules interpolate from
the densest compatible table and emit a visible non-bit-exact diagnostic. Full
legacy roots can synthesize the pack; pruned roots can validate an existing
complete pack but cannot recreate a missing exact table. Managed compact BF16,
Q8, legacy Q4, and Ref2VA packages already bundle source-bound caches.

The managed LTX 2.5 distilled pull is a public, self-contained Sawfwair
distribution accepted with `--accept-model-license`. It bundles an MLX affine
Q4/group-64 Gemma language tower plus the BF16 LTX projection and tokenizer;
users do not download the original BF16 text tower or quantize anything. The
video transformer and every non-text model component remain BF16.

Managed LTX 2.5 Distilled and Full downloads already contain BF16 transformers
in the exact native module namespace, so no post-install optimization copy is
created. For official or offline roots, `model optimize` can still stream the
source transformer payloads into `.mere-run/ltx25-native-v1` without changing
precision. `--text-encoder-only` builds the self-contained text pack under
`.mere-run/ltx25-text-q4-v1`. Compatible loads validate the pinned source
receipt and prefer a valid bundled/local Q4 pack, falling back to the official
BF16 text tower when it is absent. Use `--force` for an explicit rebuild when
source transformers are present.

### `mere.run status`

Combines the model inventory with a local API probe. It reports the active
model-store path/source, installed managed models, and whether the configured
API server answers `/health`. When the server exposes `/runtime/status`, that
snapshot (active models, admission, batching, KV reuse, cache/memory stats) is
the primary readout; the model IDs from `/v1/models` are the fallback.

### `mere.run model capabilities`

Summarizes this machine, the managed models it can run, the preferred
setup-agent tier, chat winners by RAM band, and cross-modality starter coverage.
Pass `--all` to include models that are blocked by platform or memory
requirements.

### `mere.run model info`

Shows the resolved install, source, ownership, and catalog root for one
canonical model ID. Explicit bindings without an on-disk manifest are validated
using the identity and terms acknowledgement stored in the registry.

### `mere.run model pull`

Downloads a managed model from its cataloged Hugging Face source. Pulls are
checked against the managed capability catalog before download so low-memory
machines do not fetch models they cannot run. Pass `--allow-unsupported` only
when you intentionally accept that risk or are using external hardware.

A valid model in a registered location satisfies a normal pull and prevents a
duplicate download. `--force` always installs or replaces the primary-store
copy; it never modifies the external payload.

Pass `--cache-dir PATH` to put one pull's content-addressed payload on a
specific volume. Preflight, disk estimates, downloads, and the installed links
all use that path. Disconnecting an external cache volume makes the model
unavailable. Managed replacement is transactional: the complete staged root is
validated before the old root is renamed, and a failed final validation or
alias installation restores the old root.

Access-gated models and models with material non-commercial, research-only, or
revenue-limited terms require `--accept-model-license` for restricted downloads and
never auto-download at runtime. A custom license alone does not trigger the
flag. The CLI and macOS app show the
applicable model/component terms before acceptance. Passing the flag or choosing
**Accept & Download** confirms that you reviewed and accept those terms and
agree to comply. Schema 3 of the installed `mererun_model.json` records the
immutable source revisions, term URLs, and acceptance confirmation. mere.run
does not determine whether your intended use is
permitted. The complete inventory is in
[`model-sources.md`](../model-sources.md#restricted-model-downloads).

For example, the FLUX.1-dev package is gated and noncommercial. To keep its
approximately 34 GB payload on an external volume, provide both acceptance and
the cache directory:

```bash
mere.run model pull image-flux1-dev \
  --accept-model-license \
  --cache-dir /Volumes/Models/mere-run-cache
```

Laguna XS 2.1 is released under the permissive OpenMDW-1.1 license, and its
public Hugging Face repository is not gated. It therefore installs without a
separate acceptance flag while retaining the upstream license file:

```bash
mere.run model pull text-chat-laguna-xs-2-1
mere.run text chat --model text-chat-laguna-xs-2-1 --prompt "Hello"
```

Nemotron 3.5 Lightning is also an explicit, non-automatic managed pull. Its
target and DSpark companion retain NVIDIA's OpenMDW-1.1 license files and exact
source revisions:

```bash
mere.run model pull text-chat-nemotron-35-lightning
mere.run text chat --model text-chat-nemotron-35-lightning --stats --prompt "Hello"
```

Nemotron 3 Nano Omni is a native Swift/MLX omni runtime for 112+ GB Apple
Silicon machines. It pins the official 66.06 GB BF16 snapshot and accepts text,
images, local audio, and local video, producing text. PDF documents are rendered
to page images before prompting, matching the model's actual input contract.
The explicit licensed pull and first run are:

```bash
mere.run model pull omni-chat-nemotron3-nano-30b-a3b-bf16 \
  --accept-model-license
mere.run text chat \
  --model omni-chat-nemotron3-nano-30b-a3b-bf16 \
  --image page.png \
  --prompt "Read and summarize this page."
```

The model is also available from `api serve --engine
text-chat-nemotron-omni`. OpenAI-compatible message content may contain
`image_url`, `audio_url`, and `video_url` parts. Media stays local: use local
paths or `file:` URLs for image/video payloads.

### `mere.run adapter list` and `mere.run adapter pull`

Adapters use a separate checksum-pinned catalog and install under:

```text
~/Library/Application Support/MereRun/adapters/<adapter-id>/<version>/
```

Catalog entries include the promoted Mere Platform Assistant v22 and the
gated FLUX.2-dev Turbo and remote-only Apache-2.0 LightX2V Wan 2.1 I2V
adapters:

```bash
mere.run model pull text-chat-gemma4-12b-4bit
mere.run adapter list
mere.run adapter pull mere-platform-assistant
mere.run adapter pull flux2-dev-turbo-8step --accept-license
mere.run adapter pull scail2-lightx2v-4step
mere.run adapter pull minimax-h3-lightx2v-ref2v-4step-v0.1
mere.run text chat \
  --model text-chat-gemma4-12b-4bit \
  --lora mere-platform-assistant \
  --prompt "What can you help me with in Mere?"
```

`adapter pull` downloads the immutable upstream artifact, verifies its exact
byte count and SHA-256, and prints the installed adapter path on stdout.
Catalog IDs are accepted by `text chat --lora`, `api serve --lora`, and the
compatible image and SCAIL video adapter surfaces; local paths remain
supported. `image generate --lora flux2-dev-turbo-8step` applies its published
eight-step FLUX.2-dev recipe and can be combined with repeated local
`--lora PATH[=SCALE]` values. The default `video animate --profile fast`
selects the SCAIL adapter ID and its published four-step schedule automatically.

### `mere.run setup`

Guided onboarding for the shared model store and first local agent. The command
offers a Pi-powered Mere agent, a BYOA prompt for Claude/Codex, or manual
commands. The small local agent model is the tool-capable
`text-agent-ornith-9b`; hardware-tier setup can select Gemma 4, Qwen3.6 nano on
Linux, or DeepSeek V4 Flash. On 96 GB+
Apple Silicon Macs, `text-agent-deepseek-v4-flash` is the preferred managed
setup-agent tier; smaller tool-capable native agents are alternatives, not
upgrades.
`text-code-north-mini` can be pulled, inspected, and run through the native
GGUF code runtime for direct coding comparisons against `text-code-qwen3`.
The `text-code` API lane rejects tool calls, so these models are not Pi setup
agents.
`text-agent-ornith-9b` can be pulled, inspected, and run through the native
Qwen-family MLX/OptiQ runtime for coding-agent comparisons.
Ornith 1.5 35B-A3B has four native MLX targets:
`text-agent-ornith-35b-mlx-4bit`, `text-agent-ornith-35b-mlx-6bit`,
`text-agent-ornith-35b-mlx-8bit`, and the unquantized BF16 compatibility id
`text-agent-ornith-35b-mlx`. All require an explicit `model pull`. Q6, Q8, and
BF16 pulls also install one shared, pinned BF16 MTP head from the authoritative
base checkpoint. Q4 pulls one Sawfwair packaging snapshot with its target,
vision shard, and MTP head embedded. Use `model capabilities` for the current machine's explicit
speed/balanced/quality choices. The conservative tiers are Q4 at 32 GB, Q6 at
48 GB, Q8 at 64 GB, and BF16 at 96 GB; 128 GB is recommended for BF16.
The recommended `text-agent-ornith-35b-mlx-4bit` explicit-pull lane installs
its Q4 target, authoritative base vision shard, and MTP head from one pinned
snapshot. It has a 32 GB minimum and 48 GB recommendation. The full
`vision-chat-ornith-35b` BF16 quality reference has a 96 GB minimum and
128 GB recommendation.

`text-agent-ornith-35b` is the larger GGUF Ornith eval target and runs through
the native `text-code`/llama.cpp path.

### `mere.run model remove`

Removes a managed install and its unshared backing payloads, while preserving
files referenced by another managed or legacy consumer. Confirmation and output
show referenced versus reclaimable bytes. Pass `--keep-cache` to remove only the
install links, or `--force --json` for structured automation.

For an explicit external binding, the same command unregisters the binding and
reports that the payload was preserved. Models found through an external search
root cannot be deleted individually by mere.run; unregister the root with
`model location remove`. `model storage`, `model gc`, and manifest repair own
only the primary store and Mere's Hub cache.

### `mere.run model repair-manifests`

Repairs missing manifest metadata for known models in the local store. Preview
the exact writes and consume a typed report with:

```bash
mere.run model repair-manifests --dry-run --json
mere.run model repair-manifests --json
mere.run model repair-manifests --model video-minimax-h3-fl2va-bf16-mlx \
  --accept-model-license --json
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

Reads and updates typed per-model API runtime settings that control how a model
behaves when served by `mere.run api serve`. `get` prints the stored settings
for one managed model ID or configured alias; `set` updates them.

The macOS Studio's top-level **Serving & Agents > Model Pool** view uses these
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
- `fused` — runs the versioned 24-case Lite or 110-case Comprehensive
  chat/code/tools/long-context/vision quality suite with native sampled
  profiles, repeated trials, provenance, and optional final-target logprob
  calibration diagnostics.
- `fused-fixture` — stamps or verifies normalized pinned-fixture content hashes
  without loading a model. The external source material remains non-vendored;
  `scripts/import-fused-benchmark-fixtures.py` regenerates the selected cases
  from the revision- and hash-locked source manifest.
- `tool-calls` — runs a small tool-call selection eval against local chat models.
- `tool-continuations` — evaluates Gemma 4 continuation after completed tool
  calls.
- `code` — runs a real coding-eval slice against local coding models.
- `gemma4-kv` — compares Gemma4 default key-value (KV) cache decode against
  packed PolarKV.
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

- [Model sources](../model-sources.md)
- [Configuration](../configuration.md)
- [Testing guide](../testing.md)
