# Model sources

Weights reach `mere.run` in three local-first ways:

1. **Managed pulls** — cataloged Hugging Face snapshots installed into the local
   model store by `mere.run model pull`
2. **Registered locations** — read-only search roots or explicit canonical-ID
   bindings managed by `mere.run model location`
3. **Local paths** — a directory you point at yourself with `--model`,
   `--model-root`, or the command's equivalent option

There is no private model archive, no credentialed mirror, and no central host
in between. Every managed model's source repository and pinned revision are in
the catalog, so you can see exactly where each byte came from.

The canonical local model store is:

```text
~/Library/Application Support/MereRun/models
```

Override that with `MERERUN_MODELS_DIR` or `--models-root`.

Registered locations augment the normal persisted primary store without
copying payloads or creating symlinks. Search roots use
`<root>/<canonical-model-id>/` and require each model's managed manifest.
Explicit bindings can point a canonical ID at an arbitrarily named directory;
mere.run validates the checkpoint at registration time and stores its identity
in `~/Library/Application Support/MereRun/model_locations.json` without writing
metadata into the external directory. Externally registered files remain
read-only and are outside `model remove`, `model gc`, manifest repair, and
storage-reclamation ownership.

## Canonical managed model IDs

This is the authoritative public catalog list. It is kept in sync with
`ManagedModelCatalog.allSpecs`, and the test suite fails if the table drifts
from the runtime catalog used by `mere.run model list`,
`mere.run model capabilities --all`, and `mere.run model pull`.

API-served catalog entries also own a typed profile for their task, serving
engine, input/output modalities, tool and structured-output support, reasoning
levels and mappings, context/output limits, and OpenAI compatibility behavior.
`/v1/models`, request validation, agent eligibility, and the Pi provider all
project from that profile. Runtime aliases and configured limits are applied as
an effective overlay; they are not a second capability catalog.

<!-- managed-model-catalog:start -->
| Catalog category | Model ID |
| --- | --- |
| `image` | `image-klein-nano` |
| `image` | `image-klein-max` |
| `image` | `image-klein-9b` |
| `image` | `image-klein-base` |
| `image` | `image-klein-base-9b` |
| `image` | `image-klein-shared` |
| `image` | `image-bonsai-binary` |
| `image` | `image-bonsai-ternary` |
| `image` | `image-zimage-nano` |
| `image` | `image-zimage-max` |
| `image` | `image-zimage-base` |
| `image` | `image-hidream-o1` |
| `image` | `image-hidream-o1-dev` |
| `image` | `image-sensenova-u1-5-8b-mot` |
| `image` | `image-krea2-raw` |
| `image` | `image-krea2-turbo` |
| `image` | `image-qwen-edit-2511` |
| `image` | `image-qwen-edit-2511-lightning` |
| `image` | `image-ideogram4-sdnq-uint4` |
| `text-chat` | `text-chat-mebot` |
| `text-chat` | `text-chat-psi-agent` |
| `text-chat` | `text-chat-gemma4` |
| `text-chat` | `text-chat-gemma4-turbo` |
| `text-chat` | `text-chat-gemma4-12b` |
| `text-chat` | `text-chat-gemma4-12b-4bit` |
| `vision-chat` | `vision-chat-gemma4-12b` |
| `text-chat` | `text-chat-gemma4-nano` |
| `text-chat` | `text-chat-gemma4-max` |
| `text-chat` | `text-chat-laguna-s-2-1` |
| `text-chat` | `text-chat-laguna-xs-2-1` |
| `text-chat` | `text-chat-inkling-small` |
| `vision-chat` | `vision-chat-muse-glimmer-30b` |
| `text-chat` | `text-chat-nemotron-35-lightning` |
| `omni-chat` | `omni-chat-nemotron3-nano-30b-a3b-bf16` |
| `text-chat` | `text-chat-q36-nano` |
| `vision-chat` | `vision-chat-q38-27b` |
| `vision-chat` | `vision-chat-q38-27b-4bit` |
| `vision-chat` | `vision-chat-q38-flash-next-mixed` |
| `vision-chat` | `vision-chat-q38-flash-next-3bit` |
| `vision-chat` | `vision-chat-q38-flash-next-4bit` |
| `text-chat` | `text-chat-bonsai-27b-1bit` |
| `text-chat` | `text-chat-bonsai-27b-2bit` |
| `text-code` | `text-agent-ornith-9b` |
| `text-code` | `text-agent-ornith-35b-mlx-4bit` |
| `text-code` | `text-agent-ornith-35b-mlx-6bit` |
| `text-code` | `text-agent-ornith-35b-mlx-8bit` |
| `text-code` | `text-agent-ornith-35b-mlx` |
| `text-code` | `text-agent-qwen35-9b` |
| `text-code` | `text-code-north-mini` |
| `text-code` | `text-agent-ornith-35b` |
| `text-chat` | `text-chat-q36-nano-gguf` |
| `text-chat` | `text-agent-deepseek-v4-flash` |
| `text-chat` | `text-chat-lfm25-a1b-8bit` |
| `text-chat` | `text-chat-lfm25-a1b-bf16` |
| `text-chat` | `text-chat-lfm25-1.2b-bf16` |
| `text-chat` | `text-chat-lfm25-1.2b-qad-4bit` |
| `text-chat` | `text-chat-lfm25-2.6b-4bit` |
| `text-chat` | `text-chat-lfm25-2.6b-qad-4bit` |
| `text-chat` | `text-chat-lfm25-2.6b-bf16` |
| `vision-chat` | `vision-chat-lfm25-3b-8bit` |
| `speech-tts` | `speech-tts-qwen3-nano` |
| `speech-tts` | `speech-tts-qwen3-customvoice` |
| `speech-asr` | `speech-asr-qwen3` |
| `speech-asr` | `speech-asr-parakeet` |
| `speech-diarization` | `speech-diarization-sortformer` |
| `text-code` | `text-code-qwen3` |
| `text-embed` | `text-embed-qwen3-0.6b` |
| `vision-embed` | `vision-embed-qwen3-vl-2b` |
| `text-anonymize` | `text-anonymize-privacy-filter` |
| `vision-ocr` | `vision-ocr-infinity-pro` |
| `vision-ocr` | `vision-ocr-infinity-pro-int8` |
| `vision-ocr` | `vision-ocr-lighton` |
| `vision-segment` | `vision-segment-sam31` |
| `vision-ground` | `vision-ground-falcon-perception` |
| `vision-flood` | `vision-flood-terramind-base` |
| `vision-face` | `vision-face-buffalo-l` |
| `vision-geometry` | `vision-geometry-moge2-small` |
| `vision-depth` | `vision-depth-vda-small` |
| `vision-depth` | `vision-depth-vda-small-metric` |
| `vision-geometry` | `vision-geometry-da3-small` |
| `image-3d` | `image-3d-triposr` |
| `image-3d` | `image-3d-instantmesh-base` |
| `image-3d` | `image-3d-trellis2-4b` |
| `music` | `music-acestep` |
| `music` | `music-acestep-xl-base` |
| `music` | `music-acestep-xl-sft` |
| `music` | `music-acestep-xl-turbo` |
| `music` | `music-acestep-xl-turbo-lm4b` |
| `music` | `music-acestep-lm-1.7b` |
| `music` | `music-acestep-lm-4b` |
| `music` | `music-minimax-music3` |
| `music` | `music-magenta-rt2-small` |
| `music` | `music-magenta-rt2-base` |
| `music` | `music-muscriptor-small` |
| `music` | `music-muscriptor-medium` |
| `music` | `music-muscriptor-large` |
| `music` | `music-separate-bs-roformer-viperx-1297` |
| `music` | `music-separate-bs-roformer-4stem` |
| `music` | `music-separate-mel-roformer-dereverb` |
| `music` | `music-separate-mel-roformer-denoise` |
| `audio` | `audio-enhance-ap-bwe-16kto48k` |
| `audio` | `audio-enhance-universr-audio` |
| `sfx` | `sfx-woosh-dflow` |
| `sfx` | `sfx-woosh-flow` |
| `sfx` | `sfx-woosh-clap` |
| `sfx` | `sfx-woosh-synchformer` |
| `sfx` | `sfx-woosh-vflow-8s` |
| `sfx` | `sfx-woosh-dvflow-8s` |
| `sfx` | `sfx-mmaudio-large-44k-v2` |
| `video` | `video-ltx-av` |
| `video` | `video-ltx23-av-mlx` |
| `video` | `video-ltx23-full-mlx` |
| `video` | `video-ltx23-a2vid-mlx` |
| `video` | `video-ltx25-distilled-bf16` |
| `video` | `video-ltx25-full-bf16` |
| `video` | `video-wan22-ti2v-5b-mlx` |
| `video` | `video-minimax-h3-fl2va-mlx` |
| `video` | `video-minimax-h3-fl2va-bf16-mlx` |
| `video` | `video-minimax-h3-fl2va-8bit-mlx` |
| `video` | `video-minimax-h3-fasth3-vsa-datafree-mlx` |
| `video` | `video-minimax-h3-ref2va-mlx` |
| `video` | `video-cosmos3-edge-mlx` |
| `video` | `video-scail2-14b-mlx` |
| `video` | `video-dreamx-world-5b-ar-mlx` |
| `vision-fire` | `vision-fire-terramind-base` |
| `vision-embed` | `vision-embed-tessera-v2-nano` |
| `vision-embed` | `vision-embed-tessera-v2-small` |
| `vision-embed` | `vision-embed-tessera-v2-medium` |
| `vision-embed` | `vision-embed-tessera-v2-large` |
| `vision-embed` | `vision-embed-tessera-v2-teacher` |
| `vision-embed` | `vision-embed-olmoearth-v12-nano` |
| `vision-embed` | `vision-embed-olmoearth-v12-tiny` |
| `vision-embed` | `vision-embed-olmoearth-v12-small` |
| `vision-embed` | `vision-embed-olmoearth-v12-base` |
<!-- managed-model-catalog:end -->

### Restricted model downloads

mere.run is free and open source, but model and component licenses remain the
terms of their respective owners. mere.run does not bundle these weights or
decide whether your intended use qualifies. Before downloading an access-gated
model or one with a material use limit, such as non-commercial, research-only,
or revenue-limited terms, review the listed terms and pass
`--accept-model-license`. Passing the flag and continuing with the download
confirms that you accept those terms and agree to comply with them:

```bash
mere.run model pull vision-face-buffalo-l --accept-model-license
mere.run model pull --all --accept-model-license
```

Without that flag, a single-model pull and its preflight are blocked; `--all`
skips restricted models. Restricted models never auto-download from an
inference command. The macOS app presents the same explicit acceptance before a
download and exposes the term links in its Models and Advanced views.
Batch downloads through `agent onboard --pull-recommended` skip restricted
entries without acceptance, while `open-webui quickstart --pull`
validates all configured models before downloading any; both accept the same
`--accept-model-license` flag.

| Models | Upstream terms that require acceptance |
| --- | --- |
| `image-klein-9b`, `image-klein-base-9b` | FLUX Non-Commercial License v2.1; non-commercial, non-production use |
| `image-krea2-raw`, `image-krea2-turbo` | Krea 2 Community License; commercial use is limited to entities below USD 1M trailing annual revenue, plus use/distribution conditions |
| `image-sensenova-u1-5-8b-mot` | Apache License 2.0 |
| `image-ideogram4-sdnq-uint4` | Ideogram Non-Commercial Model Agreement |
| LFM2.5 text and vision targets plus their DSpark companions | LFM Open License v1.0; commercial use by entities at or above USD 10M annual revenue is excluded |
| `vision-chat-muse-glimmer-30b` | Apache-2.0 plus Meta's bundled usage policy; upstream says the model is not intended for download or use by people under 18 |
| `vision-chat-q38-flash-next-mixed`, `vision-chat-q38-flash-next-3bit`, `vision-chat-q38-flash-next-4bit` | Qwen Community License 1.0; redistribution, attribution, and restricted-use terms |
| `vision-segment-sam31` | Meta SAM License custom use, trade-control, attribution, and redistribution conditions |
| `vision-face-buffalo-l` | InsightFace pretrained weights; non-commercial research use |
| `vision-embed-olmoearth-v12-{nano,tiny,small,base}` | OlmoEarth Artifact License; prohibited military, defense, intelligence, human-surveillance, policing, and listed extractive uses |
| `video-minimax-h3-fl2va-mlx`, `video-minimax-h3-fl2va-bf16-mlx`, `video-minimax-h3-fl2va-8bit-mlx`, `video-minimax-h3-fasth3-vsa-datafree-mlx`, `video-minimax-h3-ref2va-mlx` | MiniMax-H3 Community License; use, distribution, and display are excluded in the United States, European Union, United Kingdom, and Republic of Korea, with downstream notice and safeguard obligations |
| `image-3d-trellis2-4b` | the mounted DINOv3 encoder is gated under Meta's custom DINOv3 License |
| `music-muscriptor-{small,medium,large}` | CC BY-NC 4.0 model weights |
| `sfx-woosh-*` | CC BY-NC 4.0 Woosh or MMAudio Synchformer weights |
| `sfx-mmaudio-large-44k-v2` | CC BY-NC 4.0 MMAudio checkpoints plus Apple's research-only DFN5B encoder terms |
| `video-ltx-av`, `video-ltx23-av-mlx`, `video-ltx23-full-mlx`, `video-ltx23-a2vid-mlx`, `video-ltx25-distilled-bf16`, `video-ltx25-full-bf16` | LTX-2 Community License; entities at or above USD 10M annual revenue need a paid commercial license, plus acceptable-use conditions. The 2.3 MLX paths also install a hidden Gemma 3 text encoder under Google's Gemma Terms and Prohibited Use Policy. The 2.5 checkpoints include Gemma 4 weights under Apache License 2.0. |

The catalog pins every restricted download source to an immutable commit. New
managed installs write those repository revisions, every applicable
model/component license and URL, and the acceptance result into schema 3
of `mererun_model.json`. `mere.run model info MODEL` displays the same record.
Pre-existing installs remain usable and are not retroactively treated as an
acceptance.

The flag is not a generic click-through for every custom model license. Public,
ungated downloads whose licenses grant commercial use by exercising the
licensed rights do not require it. This includes Poolside Laguna S/XS 2.1 and
NVIDIA Cosmos3-Edge under OpenMDW-1.1, NVIDIA Sortformer under the NVIDIA Open
Model License, and the hidden Gemma 3 companion under the Gemma Terms of Use.
Their terms still apply. Managed downloads retain the available license,
README, attribution, and immutable source provenance.

`music-separate-bs-roformer-viperx-1297`,
`music-separate-bs-roformer-4stem`,
`music-separate-mel-roformer-dereverb`, and
`music-separate-mel-roformer-denoise` also do not require separate acceptance. The
pinned AEmotion Studio model release includes an explicit MIT `LICENSE`
and an MIT model-card declaration. The managed install retains both files and
admits the weights, source configuration, model card, and license only when all
four match their model-specific frozen byte counts and SHA-256 digests. The
MelBand profiles additionally retain separate dereverb and denoise source
configs and exact 913 MB checkpoint hashes.

`audio-enhance-ap-bwe-16kto48k` does not require separate acceptance. AP-BWE's
pinned source repository states that both code and pretrained weights are MIT.
The managed public transport snapshot retains the code and weights license
files, source config, and the exact official 16→48 kHz checkpoint archive;
mere.run verifies all four byte counts and SHA-256 digests before loading.

`audio-enhance-universr-audio` does not require interactive acceptance,
but its two upstream licenses must not be conflated. The native port follows
the MIT-licensed `woongzip1/UniverSR` source at its pinned commit. The separately
downloaded official `woongzip1/universr-audio` checkpoint is CC BY 4.0. The
managed install verifies the checkpoint, source configuration, and model card
by frozen revision, byte count, and SHA-256 before loading.

`image-zimage-nano` also does not require separate acceptance. Its canonical
`Tongyi-MAI/Z-Image-Turbo` base is Apache-2.0; the pinned mflux conversion's
model-card license label is inconsistent with that canonical source and is not
treated as a new restriction on the converted weights.

`speech-diarization-sortformer` installs the fp16 conversion from
`mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16` at immutable revision
`e23e6404bd9859e93edbf94a740eb1c7fc58f12e`. Its source checkpoint is NVIDIA's
`diar_streaming_sortformer_4spk-v2.1`, referenced at immutable revision
`fafaab5faa1617a0ca52d38dd3dc4bd636800d3d`; the weights are installed
separately and are not vendored in this repository.

Install most catalog IDs from managed Hugging Face sources with `mere.run model
pull`. A small number of legacy and local catalog IDs remain so
existing installs and explicit local paths keep working:

- `image-klein-shared`
- `text-chat-mebot`
- `text-chat-psi-agent`

`image-klein-shared` is an internal shared-component install shape, and the
text-chat IDs listed here remain local-path-only until they have public Hugging
Face sources.

`text-agent-qwen35-9b` is a low-memory GGUF coding model. It uses the public
Hugging Face source `unsloth/Qwen3.5-9B-GGUF` and selects
`Qwen3.5-9B-Q4_K_M.gguf`. Its `text-code` API lane does not expose tool calls;
the small Pi setup-agent tier instead uses `text-agent-ornith-9b`.

`text-code-north-mini` installs the Unsloth GGUF quant of Cohere Labs' North
Mini Code 1.0 at the pinned catalog revision. North Mini Code is a 30B total /
3B active coding MoE with a 256K advertised context window; mere.run uses the
`North-Mini-Code-1.0-UD-Q4_K_M.gguf` file so the model runs through the same
native Swift/llama.cpp `text code` path as the existing Qwen coder. It requires
a llama.cpp runtime with `cohere2moe` architecture support.

`text-agent-ornith-9b` installs the public
`sahilchachra/ornith-1.0-9b-optiq-5bpw-mlx` snapshot at the pinned catalog
revision. Ornith is a DeepReinforce agentic coding model with Qwen3.5 text
architecture metadata; mere.run treats this MLX OptiQ quant as a native
Qwen-family runtime target for `chat`, `api serve`, and setup-agent experiments.

The Ornith 1.5 native family pins the official
`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`, `-6bit`, `-8bit`, and BF16
`ornith-ai/Ornith-1.5-35B-A3B-MLX` repositories as
`text-agent-ornith-35b-mlx-4bit`, `-6bit`, `-8bit`, and
`text-agent-ornith-35b-mlx`. The 35B-parameter MoE activates about 3B
parameters per token, advertises a 262,144-token context, and runs through the
native Qwen-family runtime. Runtime auto-download remains disabled. Every pull
also installs the final safetensors shard and index containing `mtp.*` from the
pinned authoritative base checkpoint as the shared
`text-agent-ornith-35b-mtp` companion. The authoritative
[`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
base checkpoint declares the model under the MIT license in its model-card
metadata; the MLX conversion repository does not duplicate the license file.

`text-agent-ornith-35b` installs DeepReinforce's public
`deepreinforce-ai/Ornith-1.0-35B-GGUF` Q4_K_M file at the pinned catalog
revision. It runs through the native Swift/llama.cpp `text code` path for
larger Ornith coding-agent comparisons and uses a 32K runtime context by
default to keep local evaluations predictable.

`text-agent-deepseek-v4-flash` is the preferred managed setup-agent tier on
96 GB+ Apple Silicon Macs, with 128 GB recommended. It pulls the official
pure-Q2 0731 imatrix GGUF from `antirez/deepseek-v4-gguf` at an immutable
revision (80.76 GiB, SHA-256
`ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`).
Mere keeps one full-resident DS4 server, caps the operational context at 32K,
uses a 1,024-token prefill chunk, and limits disk KV checkpoints to 8 GiB.
Smaller Qwen setup agents are lower-memory alternatives, not upgrades from
DeepSeek V4 Flash. Avoid repeatedly unloading and reloading this tier under
memory pressure.

`text-chat-q36-nano` uses the public `mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`
snapshot. That Hugging Face repo includes an MTP head (`mtp.safetensors`) for
OptiQ serving; mere.run loads that draft head when present, but only uses it for
adaptive speculative decode when the effective prompt and context window are
long enough. Short-context requests decode with the main chat weights.

`vision-chat-q38-27b` installs Qwen's official `Qwen/Qwen3.8-27B` BF16
checkpoint at immutable revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. It is a dense 27B native
vision-language model with hybrid linear/full attention, a 262,144-token
native context window, image and video weights, and Apache-2.0 terms. mere.run
supports text and local-image prompts through the native Qwen-family runtime;
video input is not yet exposed. The selected snapshot is 55.59 GB, requires an
explicit `model pull`, and is cataloged for 64 GB unified memory minimum with
96 GB recommended. The pull retains the published generation and processor
metadata as well as the license.

`vision-chat-q38-27b-4bit` installs
`EigenLabs/Qwen3.8-27B-4bit` at immutable revision
`eda45ab47f465d08d6558f0353a2346e2eb9d5b3`. This is the independently
reconverted 4-bit/group-64 MLX affine target pinned by the Qwen MLX Fast track.
It retains the same Qwen3.8 text, code, image, context, and sampling contracts.
The managed pull mounts the track's matching proposal-only 4-bit/group-64 head,
`morgan/qwen38-27b-mtp-r20k-lr3-q4-g64-q2-rerank` at revision
`fd4a99c590dd6e468c0e2a28168c235e32151a4b`, under `mtp/`. The companion uses
a proposal-only 2-bit compact readout with exact 4-bit shortlist reranking.
Because the optimized target doesn't contain a vision tower, the managed pull
also mounts the official Qwen3.8 configuration, processor metadata, weight
index, and the first indexed BF16 shard under `vision/` from the pinned official
revision above. That shard contains all 333 vision tensors; language tensors in
it are not loaded. Qwen's official Apache-2.0 license is retained separately.
The sources total about 19.49 GB. Greedy MTP decode is enabled by default for
this quantized target because its small-batch Q4 verification is serial-exact;
set `MERERUN_Q35_MTP_SPECULATION=0` to use target-only decode.

`vision-chat-q38-flash-next-mixed` installs Sawfwair's immutable
`Qwen3.8-Flash-Next-MLX-Mixed-2bit` revision
`0bdf3edf02df271e9898f17a7882e5e6a8feb58a`. It applies affine Q2/group-128
to the 48 routed-expert banks, Q4 to eligible core matrices and the n-gram
table, and retains embeddings, QSA indexers, routers, vision, and selected MTP
paths in BF16. Its 73.10 GB of weights are the managed 128 GB Mac profile,
with 112 GB minimum and 128 GB recommended.

`vision-chat-q38-flash-next-3bit` installs Sawfwair's immutable
`Qwen3.8-Flash-Next-MLX-Activation-3bit` revision
`c699bd611366cbc441377275bd1b7a6d2e18e1df`. The 48 routed-expert banks use fresh
BF16-to-Q3/group-64 codes with image and text activation-weighted scale and
bias refits. Eligible core, MTP, and vision matrices remain Q4. The n-gram
table remains Q4/group-32. The 89.67 GB managed pull is cataloged for 96 GB
minimum and 128 GB recommended.

The native qualification on a 128 GiB Apple Silicon host passed 58 of 61 cases
by exact expected output. The three misses matched the pinned Q4 output. The
sealed 16-case holdout produced identical Q3 and Q4 outputs, including exact
output on all eight image and OCR cases. The holdout peak was 62.1 GB for Q3
and 77.2 GB for Q4, with no swap growth. These bounded checks establish Q4
no-regression for the tested cases, not general model quality or BF16 parity.

`vision-chat-q38-flash-next-4bit` installs the companion immutable
`Sawfwair/Qwen3.8-Flash-Next-MLX-4bit` revision
`6cc9bbc0fae9ce26b7670b3ed1e26d557c154506`. Its 104.74 GB of weights use
affine Q4/group-64 for eligible matrices, with Q4/group-32 for the 160-wide
n-gram table. It is cataloged for 160 GB minimum and 192 GB recommended rather
than as a 128 GB alternative.

All three artifacts include the Qwen Community License 1.0 and a complete
source/output hash receipt. Pulls are explicit and require either
`--accept-model-license` or its equivalent `--accept-license-terms` alias. The
native Qwen4Exp runtime supports text generation; image input remains
unqualified. Version 0.46.0 and later implement the trained QSA micro-block
selector beyond 2,048 tokens, replacing v0.45.0's prompt-plus-generation
cap. Up to the indexer budget, standard causal attention is exact because every
visible block is selected. Beyond it, each query selects 512 complete four-token
blocks and includes the current partial block. Future blocks are excluded
before selection. The same QSA path is used by the bundled one-layer MTP head,
with exact target verification of proposed tokens.

`--context-size` bounds prompt plus generation up to the checkpoint's published
262,144-token limit. This is an architectural limit, not a 262k memory or quality
qualification on a 128 GB Mac. KV and MTP history still grow with context even
though sparse-attention temporaries are tiled; an explicit `--context-size 32768`
keeps the requested window smaller. Existing model downloads already contain
the indexer weights, so no new quantization or download is needed.

The mixed checkpoint measured 1.43–1.78x faster MTP decode in the original
short-context real-checkpoint comparisons; these are not long-context timings.
Set `MERERUN_Q35_MTP_SPECULATION=0` to retain target-only decode for comparison
or memory pressure.

`text-chat-bonsai-27b-1bit` and `text-chat-bonsai-27b-2bit` install Prism ML's
public `prism-ml/Bonsai-27B-mlx-1bit` and
`prism-ml/Ternary-Bonsai-27B-mlx-2bit` snapshots at exact catalog revisions.
They are dense Qwen3.6 27B vision/reasoning models with packed affine 1-bit or
2-bit language weights and dense vision weights. The snapshots are
approximately 5.13 GB and 8.52 GB respectively and advertise a 262,144-token
context. mere.run uses the native Qwen-family text and vision runtime, native
low-bit linear and embedding kernels, and the published thinking and sampling
defaults. Both models are Apache-2.0 licensed; managed pulls retain upstream
license and notice files.

`text-chat-q36-nano-gguf` installs the Unsloth
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` quant. It is the llama.cpp/GGUF companion to
the Apple Silicon MLX `text-chat-q36-nano` path and is the default chat model
for Linux CUDA hosts.

`vision-chat-muse-glimmer-30b` pins Sawfwair's 21.38 GB selective MLX Q4
artifact at revision `6532e898dc5c1a55b51b1b108cd36728b79be751`. Its conversion
receipt pins Meta's Apache-2.0 Muse Glimmer 30B BF16 source at revision
`f84ecc3a0ea984a4c04542a84269e3d065350a6e`, records every source and output
hash, and retains Meta's `LICENSE` and `USAGE_POLICY.md`. The managed artifact
is never downloaded implicitly. The native Swift/MLX runtime implements its
52-layer local/local/local/global text stack,
NoPE global layers, gated attention, 50-layer perception encoder, interleaved
image tokens, ATEM tool calls, and low/medium/high/xhigh reasoning-strength
prompt contract. Its image path uses the released uint8 Lanczos resize behavior
and float32 position interpolation. The released artifact and offline converter
use selective Q4/group-64 over 420 text/output/adapter matrices while retaining
the token
embedding and complete vision tower in BF16; pass `--quantization-scope compact`
to quantize all 721 eligible matrices for an explicit lower-memory experiment.
The runtime discovers quantized modules from their `.scales` arrays, so both
layouts use the same inference implementation. The same pull installs z-lab's
5.54 GB DFlash2 assistant as the hidden managed model
`vision-chat-muse-glimmer-30b-dflash2`, pinned to
`z-lab/Muse-Glimmer-30B-DFlash2@b54ffdd11fa9cfe2af370012e5763d492c904128`.
The earlier 5.11 GB assistant from `meta-models/Muse-Glimmer-30B-assistant` at
revision `2c86316d689027b91123638739743fef1d425233` remains a compatible fallback
for existing installations. Pulls require explicit review and acceptance of
the target's bundled `LICENSE` and `USAGE_POLICY.md`; upstream says the model is
not intended for download or use by people under 18. Python conversion scripts
are offline artifact tooling only and are not part of inference.

`text-chat-nemotron-35-lightning` pins Sawfwair's native MLX conversion at
revision `6699e5fd3f0c5b392bb3f8bac2443276bb41958a`, produced from NVIDIA's
`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` source revision
`e0b753dc24903ad4d62f5696077da22020eca89a`. The receipt records the 52 pinned
source shards and every output hash. It repacks the released ModelOpt NVFP4
nibbles bit-for-bit into MLX's native container, retains both block and global
scales, and materializes the 46 released FP8 projections as BF16 without a
second quantizer. Managed pulls also install the separate Sawfwair MLX
conversion of NVIDIA's 967M-parameter DSpark companion at artifact revision
`d30f0914d6bbb6da36302bd9228f92824901e675`, pinned from source revision
`e3af76fbff445ef795958bee96bc1126af70fd57`. Both artifacts retain NVIDIA's
OpenMDW-1.1 license and upstream model cards, never download implicitly, and do
not require an additional mere.run acceptance gate solely because the public
repositories use a custom license identifier.

`omni-chat-nemotron3-nano-30b-a3b-bf16` stages NVIDIA's official
`Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` snapshot at immutable revision
`24e67ea000b7c2837fc8f9488aa2008524fac8ba`. The 66.06 GB checkpoint accepts
text, images, audio, and video and produces reasoning text. Its document
examples render PDF pages to images before prompting; PDF is not a separate raw
input modality. The catalog pins all 17 BF16 weight shards, typed
Nemotron-H/C-RADIO v4-H/Parakeet configuration contracts, tokenizer and media
processor metadata, NVIDIA Open Model Agreement terms, 131,072-token composed
context, and 20,480-token maximum output.

The native Swift/MLX runtime implements the C-RADIO vision tower, Parakeet
audio tower, BF16 Nemotron-H language backbone, multimodal prefill, local video
sampling, and media-aware OpenAI chat completions. It is supported on Apple
Silicon machines with at least 112 GB unified memory and recommended at 128 GB.
The model remains an explicit pull because its NVIDIA terms must be accepted:

```bash
mere.run model pull omni-chat-nemotron3-nano-30b-a3b-bf16 \
  --accept-model-license
mere.run text chat \
  --model omni-chat-nemotron3-nano-30b-a3b-bf16 \
  --video clip.mp4 \
  --prompt "Summarize the clip."
```

The runtime consumes the 17 upstream BF16 shards in place and creates a
source-bound stacked expert cache. A lightweight managed-model wrapper may
point at an immutable snapshot on external storage, avoiding a second copy of
the checkpoint while keeping it visible to `model list`, `text chat`, and
`api serve`.

`text-chat-gemma4-12b` and `vision-chat-gemma4-12b` share Google's dense Gemma
4 12B-it checkpoint; the text id uses the native chat path, while the vision id
enables OpenAI image content parts through `api serve`. Pulling either managed
12B id also pulls the companion `google/gemma-4-12B-it-assistant` MTP drafter.
The native Swift runtime uses that assistant only for greedy decode-tail
speculation after text, image, or audio prefill has produced target hidden state
and shared KV; raw local model paths and sampled generations fall back to
baseline decode. `text-chat-gemma4` is the dense bf16 Gemma 4 31B alias and is
gated for larger machines. On 32 GB Apple Silicon Macs, use
`text-chat-gemma4-turbo`, which installs the MLX NVFP4 Gemma 4 26B-A4B-it MoE
snapshot and runs through the native Swift Gemma runtime. On smaller supported
machines, `text-chat-gemma4-12b-4bit` installs the
checksum-pinned `Sawfwair/gemma-4-12B-it-MLX-4bit` snapshot as the compact
managed Gemma 12B chat tier. Sawfwair's package is independently converted
from Google's verified dense checkpoint, includes `MERERUN_CONVERSION.json`
with source and emitted artifact hashes, and is published as the user-facing
`v1.0.0` release.

`text-chat-lfm25-a1b-8bit` uses the public
`LiquidAI/LFM2.5-8B-A1B-MLX-8bit` snapshot at the pinned catalog revision. It is
a text-only MLX 8-bit directory-root model with `config.json`,
`tokenizer.json`, `tokenizer_config.json`, and sharded `*.safetensors` weights.
mere.run runs it through the native Swift LFM2 runtime; no Python bridge is used.
This quantized target does not attach the BF16-trained DSpark assistant.

`text-chat-lfm25-a1b-bf16` uses `LiquidAI/LFM2.5-8B-A1B` at revision
`b9aebfcbe28b6cb374042f495d733037550ab146`, the exact target for the managed
8B-A1B DSpark companion.

`text-chat-lfm25-1.2b-bf16` uses
`LiquidAI/LFM2.5-1.2B-Instruct` at revision
`df58c174f05ff733f83f8cae10ea9298224c8006`. Its matching
`text-chat-lfm25-1.2b-dspark` companion uses
`LiquidAI/LFM2.5-1.2B-Instruct-DSpark` at revision
`4876d04848e15a6fd48d7c1481110e7cf5d62621`.

`text-chat-lfm25-1.2b-qad-4bit` uses
`Sawfwair/LFM2.5-1.2B-Instruct-QAD-MLX-4bit` at revision
`b1caaa502c5bfd975f5219162eaa90b1e7cc7839`. The conversion pins LiquidAI's
QAD Q4_0 GGUF at revision `afbd8eaeab5dd94ba0b079ebfb02517d19641e38`,
keeps every projection nibble and scale exact in MLX group-32 affine tensors,
and records all source and output hashes in `MERERUN_CONVERSION.json`.

`text-chat-lfm25-2.6b-4bit` uses the pinned `4bit/` partition of
`LiquidAI/LFM2.5-2.6B-MLX`. The managed pull selects only that partition plus
the repository license and model card, then normalizes the nested directory
for the native dense `Lfm2ForCausalLM` runtime. The checkpoint uses affine
4-bit linear weights with a 6-bit tied embedding and is approximately 1.60 GB.
It remains target-only because the managed DSpark assistant was trained for the
unquantized target.

`text-chat-lfm25-2.6b-bf16` uses `LiquidAI/LFM2.5-2.6B` at revision
`a334ee78cd38458bb71eda24109ac42dcec1309d`, the exact target for the managed
2.6B DSpark companion.

`text-chat-lfm25-2.6b-qad-4bit` uses
`Sawfwair/LFM2.5-2.6B-QAD-MLX-4bit` at revision
`e829e540629759fdb886cb529e4d03640a11ffa7`. Its conversion pins LiquidAI's
QAD Q4_0 GGUF at revision `f4a289c8a200a5ca71005ba7abc2dad33058a450`
and follows the same exact projection conversion contract. The standard 2.6B
MLX 4-bit checkpoint remains the faster native MLX lane; the QAD variant exists
for the quantized-accuracy tradeoff. Liquid's published QAD throughput values
reuse native llama.cpp Q4_0 throughput because QAD and PTQ share that format,
not because QAD itself adds a runtime speedup.

The three exact BF16 LFM2.5 text targets install matching DSpark sidecars from LiquidAI:
`LFM2.5-8B-A1B-DSpark` at revision
`5b285c827912834665b1915f171897e49ff0f388`,
`LFM2.5-2.6B-DSpark` at revision
`458cedab07d0f7b2b05700c77e1aa463d43d6f04`, and the 1.2B revision above.
Each sidecar is roughly 0.59-0.66 GB and is loaded only beside its compatible
target. LiquidAI reports mean M4 Max speedups of 2.54x for 1.2B, 2.27x for
2.6B, and 1.18x for 8B-A1B; these are upstream measurements, not mere.run
benchmark results.
See LiquidAI's [LFM2.5-DSpark release and methodology](https://www.liquid.ai/blog/lfm2.5-dspark)
for the upstream benchmark protocol and framework integrations.

`vision-chat-lfm25-3b-8bit` uses the public
`LiquidAI/LFM2.5-VL-3B-MLX-8bit` checkpoint at revision
`4065d2c056a9c54d44fec67cf651812b55c6673f`. The managed snapshot is
approximately 3.74 GB and includes the dense LFM2.5 2.6B language backbone,
SigLIP2 NaFlex vision tower, multimodal projector, tokenizer, chat template,
and `processor_config.json`. mere.run processes local file paths and base64
data URLs natively, expands each `<image>` placeholder to the downsampled
patch grid, and continues generation through the shared LFM2 decode engine.
Remote image URLs are not fetched by the local runtime.

Useful environment variables for that path:

- `MERERUN_HUB_CACHE`: override the native Hugging Face snapshot cache path

`image-klein-9b` installs the ungated `mlx-community/FLUX.2-klein-9B` mirror for
larger distilled Klein generation and reference-image workflows. It is distinct
from `image-klein-base-9b`, which pulls the undistilled Base 9B transformer plus
shared 9B components for higher-capacity LoRA training and research workflows.

`image-bonsai-binary` and `image-bonsai-ternary` map to PrismML Apple Silicon
Bonsai Image snapshots:

- `prism-ml/bonsai-image-binary-4B-mlx-1bit`
- `prism-ml/bonsai-image-ternary-4B-mlx-2bit`

The snapshot uses a FLUX.2 Klein transformer, but its component names are
upstream-specific. Managed or local roots are expected to contain:

- `manifest.json`
- `tokenizer/tokenizer_config.json`
- `text_encoder-mlx-4bit/config.json`
- `text_encoder-mlx-4bit/model.safetensors` or `model.safetensors.index.json`
- `transformer-packed-mflux/config.json`
- `transformer-packed-mflux/quantization_config.json`
- `transformer-packed-mflux/diffusion_pytorch_model.safetensors`
- `vae/config.json`
- `vae/diffusion_pytorch_model.safetensors`
- `scheduler/scheduler_config.json`

The binary manifest records the transformer as 1-bit g128 Prism packed affine
weights; the ternary manifest records 2-bit g128 MLX packed affine ternary
weights. Both keep the text encoder in the upstream 4-bit MLX layout and run
generation through the native Swift FLUX.2 Klein pipeline with four steps, CFG
1.0, and sigma shift 3.0 by default. The binary runtime path uses native packed
1-bit affine matmul kernels on Metal and Linux CUDA, with a dequantized MLX
fallback for non-GPU or unsupported shapes.

`vision-segment-sam31` packages the native SAM 3.1 segmentation and tracking runtime used by `mere.run vision segment`, `mere.run vision track`, and `mere.run vision track-live`. Managed or local SAM roots are expected to contain:

- `config.json`
- `model.safetensors` or `model.safetensors.index.json`
- `tokenizer/tokenizer.json`
- `tokenizer/tokenizer_config.json`

Tokenizer files are technically optional for geometry prompts, but managed
installs include them because text prompts are the preferred segmentation path.
The managed package mounts tokenizer assets from the SAM 3.1 mirror while the
native MLX weights come from `mlx-community/sam3.1-bf16`.

The manifest for this package advertises both `vision_segmentation` and
`vision_tracking` capabilities.

`image-hidream-o1` and `image-hidream-o1-dev` map to the public HiDream O1
image checkpoints:

- `HiDream-ai/HiDream-O1-Image`
- `HiDream-ai/HiDream-O1-Image-Dev`

HiDream O1 uses a unified pixel-transformer root layout rather than a
VAE/text-encoder component tree. Managed or local roots are expected to contain:

- `config.json`
- `tokenizer_config.json`
- `tokenizer.json` or `vocab.json` plus `merges.txt`
- `preprocessor_config.json`
- `model.safetensors` or `model.safetensors.index.json`

The native Swift runtime validates this layout, decodes the typed root
configuration, prepares text/reference sample metadata, and runs generation
through the downloaded Qwen3-VL decoder, vision tower, timestep embedder, patch
embedder, generation-aware attention mask, and HiDream pixel head. Text-only
generation, one-reference instruction editing, and multi-reference subject
personalization share the same native path; reference modes additionally run
Qwen3-VL vision preprocessing and replace chat-template image placeholders
before denoising.

Runtime defaults come from the managed manifest:

- `image-hidream-o1-dev`: 28 steps, CFG 0.0, fixed flash FlowMatch schedule
- `image-hidream-o1`: 50 steps, CFG 5.0, shifted Flow UniPC schedule

Both checkpoints are large BF16 unified-transformer roots, about 33 GiB on disk
each before filesystem compression effects. Expect high unified-memory pressure
and prefer one-step smokes before full-quality 28/50 step runs.

`image-sensenova-u1-5-8b-mot` maps to the official
`sensenova/SenseNova-U1.5-8B-MoT` checkpoint at the exact revision recorded in
the managed catalog. The corresponding reference implementation is published
at `OpenSenseNova/SenseNova-U1`. The root contains the typed configuration,
Qwen tokenizer assets, a safetensors index, and 13 shards.

SenseNova uses one unified checkpoint rather than separate text-encoder and VAE
directories. The native runtime mirrors the checkpoint's understanding and
generation experts, caches the prompt/reference prefix, embeds RGB patches,
applies the published shifted Euler flow schedule, and decodes predicted RGB
pixels with the checkpoint's convolutional pixel head. Text-to-image and
multi-reference instruction editing share this Swift/MLX path. The managed
defaults are 50 steps, CFG 4.0, and timestep shift 3.0.

`image-krea2-raw` and `image-krea2-turbo` map to Krea's public Krea 2
checkpoints:

- `krea/Krea-2-Raw`
- `krea/Krea-2-Turbo`

Managed or local roots are expected to contain the component Diffusers layout:

- `model_index.json`
- `tokenizer/tokenizer_config.json`
- `tokenizer/tokenizer.json`
- `text_encoder/config.json`
- `text_encoder/model.safetensors`
- `transformer/config.json`
- `transformer/diffusion_pytorch_model.safetensors.index.json`
- `transformer/diffusion_pytorch_model-*.safetensors`
- `vae/config.json`
- `vae/diffusion_pytorch_model.safetensors`
- `scheduler/scheduler_config.json`

Krea also publishes root-level `raw.safetensors` and `turbo.safetensors`
transformer copies for the official codebase. Managed pulls intentionally
exclude those files and pull the split transformer component instead, so the
model store does not download the same large transformer payload twice.

The native Swift runtime follows the public Krea 2 sampler shape: Qwen3-VL text
conditioning with the Krea system prefix, layer-selected hidden-state fusion,
single-stream MMDiT denoising, 16-pixel image-token alignment, FlowMatch Euler
steps, and Qwen Image VAE decoding. The wired public generation mode is
text-to-image with optional LoRA adapters; reference images and image-to-image
are not supported for this family yet. LoRA training uses `image-krea2-raw`;
inference uses `image-krea2-turbo`.

Krea publishes sample LoRA adapters such as
`krea/Krea-2-LoRA-retroanime` and `krea/Krea-2-LoRA-kidsdrawing`. Those
adapters are trained on Raw and loaded on Turbo with Diffusers
`lora_A` / `lora_B` keys. The native loader preserves Krea's `img_in`,
`txt_in`, `text_fusion`, `time_embed`, `time_mod_proj`,
`transformer_blocks`, and `final_layer` module names so those published
adapters can be used as compatibility references.

Runtime defaults come from the managed manifest:

- `image-krea2-raw`: 52 steps, CFG 3.5, FlowMatch shift/mu 1.15
- `image-krea2-turbo`: 8 steps, CFG 0.0, FlowMatch shift/mu 1.15

The component install is about 36 GiB before filesystem compression effects.
Expect high unified-memory pressure and prefer 96 GB+ Apple Silicon machines.

`image-qwen-edit-2511` is the immutable quality lane for Qwen Image Edit 2511.
It pulls `Qwen/Qwen-Image-Edit-2511` at revision
`6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`; it does not replace or redirect
the legacy `qwen-image-edit` install. Its managed defaults are 40 steps and
true CFG 4.0 with a blank negative prompt.

`image-qwen-edit-2511-lightning` uses the same pinned base plus
`lightx2v/Qwen-Image-Edit-2511-Lightning` at revision
`d74eba145674fd7e31b949324e148e21e7118abd`. The runtime accepts only the
four-step BF16 adapter named
`Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`, verifies its
849,608,296-byte size and SHA-256
`22226e8d05d354bb356627d428809f5afd7819399b077238a2b70a82883a904f`, and
requires exactly four steps with CFG disabled.

Both managed IDs use the native 2511 edit path. It accepts one to three ordered
images across `--input` followed by repeated `--ref-image` arguments. Each
reference retains its own aspect ratio for Qwen2.5-VL semantic conditioning and
VAE appearance conditioning. The output starts from pure noise; packed
reference latents remain conditioning tokens rather than being blended into
the output latent. These two pinned lanes retain the base encoder and
transformer in BF16 at runtime; the legacy `qwen-image-edit` compatibility path
keeps its existing runtime Q4 behavior.

```bash
mere.run model pull image-qwen-edit-2511
mere.run image generate \
  --model image-qwen-edit-2511 \
  --input ./primary.png \
  --ref-image ./style.png \
  --prompt "Apply the palette of Picture 2 while preserving Picture 1's layout" \
  --output ./edited.png
```

The pinned base is about 53.75 GiB before filesystem compression; the Lightning
lane adds about 810 MiB. These sizes and the current runtime's memory behavior
require physical Apple Silicon qualification before publishing a supported
memory class or performance claim.

`image-ideogram4-sdnq-uint4` maps to WaveCut's public SDNQ uint4 Ideogram 4
snapshot:

- `WaveCut/ideogram-4-sdnq-uint4`

Managed or local roots are expected to contain:

- `model_index.json`
- `quantization_manifest.json`
- `tokenizer/tokenizer_config.json` or `tokenizer/tokenizer.json`
- `text_encoder/config.json`
- `transformer/config.json`
- `transformer/diffusion_pytorch_model.safetensors`
- `unconditional_transformer/config.json`
- `unconditional_transformer/diffusion_pytorch_model.safetensors`
- `vae/config.json`
- `vae/diffusion_pytorch_model.safetensors`
- `scheduler/scheduler_config.json`

The managed manifest records SDNQ asymmetric uint4 weights and the separate
positive and unconditional transformer branches used by Ideogram 4 guidance.
The native runtime can pull, inspect, validate, and decode SDNQ uint4
linear, embedding, and Conv2d weights, build Qwen3-VL concatenated text
features, pack Ideogram 4 text/image samples, run positive/unconditional CFG
denoising, and decode PNG output through the Flux2-style VAE. Text-to-image
`image generate` is wired; image-to-image, reference inputs, and LoRA are still
unsupported for this family.

## Hugging Face cache

Hub snapshots use the shared cache location managed by the runtime. Override it
when you want large models on an external disk:

```bash
export MERERUN_HUB_CACHE=/Volumes/Models/huggingface
swift run mere.run model pull image-zimage-nano
```

Model pulls are resumable at the file level through the Hugging Face snapshot
cache. The CLI writes a managed-model symlink from the mere.run model store to
the prepared snapshot when needed.

## Hardware support checks

Managed pulls are gated by the local capability catalog before any download. The
check uses supported local runtimes plus memory thresholds for each model
family, then blocks models that are unlikely to run reliably on this machine.

Inspect the local recommendation first:

```bash
swift run mere.run model capabilities
swift run mere.run model capabilities --all
```

If you are intentionally testing an unsupported setup, pass
`--allow-unsupported` to `mere.run model pull`.

## Model store behavior

The CLI resolves models in this order:

1. `--models-root` process override
2. `MERERUN_MODELS_DIR`
3. persisted local model-store setting
4. default `~/Library/Application Support/MereRun/models`

Examples:

```bash
# Pull into the default model store
swift run mere.run model pull image-zimage-nano

# Pull into a custom SSD-backed store
MERERUN_MODELS_DIR=/Volumes/Models swift run mere.run model pull text-chat-q36-nano
MERERUN_MODELS_DIR=/Volumes/Models swift run mere.run model pull text-chat-lfm25-a1b-8bit --accept-model-license
MERERUN_MODELS_DIR=/Volumes/Models swift run mere.run model pull text-chat-lfm25-a1b-bf16 --accept-model-license
MERERUN_MODELS_DIR=/Volumes/Models swift run mere.run model pull text-chat-lfm25-1.2b-bf16 --accept-model-license
MERERUN_MODELS_DIR=/Volumes/Models swift run mere.run model pull text-chat-lfm25-1.2b-qad-4bit --accept-model-license
MERERUN_MODELS_DIR=/Volumes/Models swift run mere.run model pull text-chat-lfm25-2.6b-4bit --accept-model-license
MERERUN_MODELS_DIR=/Volumes/Models swift run mere.run model pull text-chat-lfm25-2.6b-bf16 --accept-model-license
MERERUN_MODELS_DIR=/Volumes/Models swift run mere.run model pull text-chat-lfm25-2.6b-qad-4bit --accept-model-license
MERERUN_MODELS_DIR=/Volumes/Models swift run mere.run model pull vision-chat-lfm25-3b-8bit --accept-model-license

# Inspect installed models
swift run mere.run status
swift run mere.run model list
swift run mere.run model info image-klein-max
```

## Music, SFX, and video layouts

Some retained surfaces have more structure than a flat model root.

### ACE-Step 1.5 models

The top-level model roots are:

```text
.../models/music-acestep
.../models/music-acestep-xl-turbo
.../models/music-acestep-xl-turbo-lm4b
.../models/music-acestep-xl-sft
.../models/music-acestep-xl-base
.../models/music-acestep-lm-1.7b
.../models/music-acestep-lm-4b
```

Those roots may contain:

- `acestep-v15-turbo/`
- `acestep-v15-xl-turbo/`
- `acestep-v15-xl-sft/`
- `acestep-v15-xl-base/`
- `acestep-5Hz-lm-1.7B/` or another supported LM subdirectory
- `acestep-5Hz-lm-4B/`
- `Qwen3-Embedding-0.6B/`
- `vae/`

Older local installs that still use `music-acestep-v15-turbo/` remain
supported.

`music-acestep-xl-turbo` pulls the ACE-Step 1.5 XL turbo DiT into
`acestep-v15-xl-turbo/` and reuses the base ACE-Step 1.5 VAE and Qwen3 text
encoder components. `music-acestep-xl-turbo-lm4b` adds the optional
`acestep-5Hz-lm-4B/` 5 Hz LM. The independently pullable
`music-acestep-lm-1.7b` and `music-acestep-lm-4b` planner models can be paired
with any ACE-Step DiT through `--lm-model`; no shared-directory symlink is
required. The 1.7B planner is the upstream default. Local component discovery
therefore prefers `acestep-5Hz-lm-1.7B/` when both sizes are present; select
`--lm-model music-acestep-lm-4b` only when you explicitly want the optional 4B
planner.
`music-acestep-xl-sft` and `music-acestep-xl-base` install the non-distilled
XL checkpoints and use continuous flow sampling with CFG, APG, or ADG. Base is
also the checkpoint family for extract, lego, and complete tasks. Every
component is downloaded at the immutable revision listed in
[ACE-Step validation](./runtime/acestep-validation.md).

`mere.run music generate`, `music analyze`, and `music serve` first look for a
planner in the selected checkpoint root, then reuse an installed standalone or
`music-acestep` 1.7B planner, and finally pull `music-acestep-lm-1.7b` when LM
planning is required. Override that resolution with `--lm-model` or the legacy
same-root `--lm-subdirectory`.

### `music-minimax-music3`

MiniMax Music 3 comes from `MiniMaxAI/MiniMax-Music3` at immutable revision
`bd348f9c49ea3c1b39f33ace3436f8fad435f24e`. The managed pull selects the 25
runtime files needed by the native Swift/MLX implementation and omits the
duplicate `qwen_7B/` tree and Python-only examples. The selected snapshot is
28,517,620,807 bytes (approximately 26.6 GiB):

```text
.../models/music-minimax-music3
├── condition_encoder/
├── language_model/
├── rvq_depth_decoder/
├── scheduler/
├── tokenizer/
├── transformer/
├── vocoder/
├── config.json
├── modular_model_index.json
└── LICENSE
```

The checkpoint uses the MiniMax-Music3 Community License rather than an OSI
open-source license. Managed pulls therefore require
`--accept-model-license`, runtime auto-download is disabled, and applications
using the model must preserve the upstream attribution and usage restrictions.
Review the pinned `LICENSE` before deploying or redistributing the weights.

`mere.run music generate --model music-minimax-music3` assembles the released
caption-and-lyrics prompt contract, generates 25 Hz semantic plus residual RVQ
codes, runs the overlap-aware flow transformer, and writes native 44.1 kHz
stereo WAV output through the released vocoder. `--max-frames` exposes the
upstream 1...9,000-frame contract directly, while `--duration` converts seconds
at 25 Hz. The model card describes supported-quality generation through five
minutes; 9,000 frames is the six-minute runtime hard limit.

`--minimum-duration` and `--min-frames` add an EOS floor; duration floors round
up across the vocoder's whole 512-sample hops so the decoded WAV does not
undershoot. `--performance-mode optimized` is the BF16 default and uses a
compact semantic projection, a full-vocabulary sampling view, fused global
language-model projections, and batched flow guidance. The eight-token residual
depth prefix is recomputed without projection fusion because its cached/fused
BF16 path changed seeded codebook choices and caused lyric drift. The opt-in
`q8-lm` and `q4-lm` apply group-64 affine quantization only to the global
language model while retaining the residual-depth decoder in BF16. They are
the preferred quantized experiments because depth codebooks directly affect
vocal detail. Legacy `q8` and `q4` still quantize both autoregressive
components for compatible maximum compression. Flow stays BF16 because
installed-checkpoint timing showed its quantized kernels regress.

`--compose` runs a local native Gemma4 or Qwen-family chat model in two
constrained-JSON passes before loading MiniMax: a bar-aware song blueprint,
then lyrics and the checkpoint's three-part structured caption. Supplied lyrics
remain authoritative. The typed composition receipt and lyric-duration
preflight are stored separately and in schema 7 generation provenance. The
recipe and schema 3 profile record resolved language-model and depth-decoder
precision independently.
Composer weights are unloaded before music inference. The default lyric
preflight warns because `--duration` is an upper bound rather than a promise;
`strict` rejects sparse or structurally invalid inputs without silently adding
an EOS floor.

Euler, full autoregressive CFG, and full flow CFG remain the parity defaults.
The opt-in `--flow-solver ab2`, `--ar-cfg-frames`, and `--flow-cfg-end`
controls support reproducible full-song A/B experiments and are recorded in
recipes and profiles. They are experimental controls, not admission claims.

Generation defaults to `--memory-mode staged`, which releases the language,
flow, and vocoder weights between stages. `--memory-mode resident` keeps the
entire stack loaded for repeated work. Staged mode moves the catalog floor to
32 GB unified memory; 64 GB remains recommended for practical song lengths.
`--sample-rate 32000` produces the same stereo PCM rate exposed by the upstream
SGLang speech route; 44,100 Hz remains the native CLI default. `music serve
--model music-minimax-music3` exposes the non-streaming `/v1/audio/speech`
request shape with `input`, `instructions`, `seed`, and `max_new_tokens`, plus
explicit native duration, step, guidance, solver, CFG-cutoff, lyric-preflight,
and sample-rate controls.

MiniMax publishes its optional `music-caption-rewriter` agent skill separately
in the official
[MiniMax-Music-3 repository](https://github.com/MiniMax-AI/MiniMax-Music-3/tree/91410fb657c007ae57c60df8240f5ece5be089c7/music-caption-rewriter).
It is not part of the checkpoint snapshot or this distribution. The
model-specific `mere.run guide music generate --model music-minimax-music3`
shows the official install command, the reviewed source commit, and the
complete raw Diffusers parameter mapping.

### `music-magenta-rt2-small` and `music-magenta-rt2-base`

Magenta RT2 models use exported runtime assets from
`google/magenta-realtime-2` at revision
`010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc`. The managed pull keeps only the
files needed by the native runtime:

```text
.../models/music-magenta-rt2-small
├── models/mrt2_small/mrt2_small.mlxfn
├── models/mrt2_small/mrt2_small_state.safetensors
├── resources/musiccoca/
└── resources/spectrostream/
```

The base model uses `models/mrt2_base/` with matching `mrt2_base` filenames.
Raw `checkpoints/*.safetensors` files are not a complete `mere.run` layout.

`mere.run music generate --model music-magenta-rt2-small` renders an offline
WAV. `mere.run music realtime --model music-magenta-rt2-small` plays on the
default macOS audio device and can capture to WAV with `--output`.

### `music-muscriptor-small`, `music-muscriptor-medium`, and `music-muscriptor-large`

MuScriptor checkpoints come from the gated `MuScriptor/muscriptor-{size}`
Hugging Face repositories. Each managed root contains `config.json` and
`model.safetensors`. The weights are CC BY-NC 4.0 and require accepting the
upstream access terms before `mere.run model pull` can download them.

`mere.run music transcribe` decodes input audio to mono 16 kHz, runs the exact
published five-second HTK mel frontend and native MLX transformer, then writes
a format-1 MIDI file with one track per detected instrument. JSON and JSONL
event output are also available.

### `sfx-woosh-dflow`

The Woosh DFlow model root is:

```text
.../models/sfx-woosh-dflow
└── checkpoints/
    ├── Woosh-DFlow/
    ├── Woosh-AE/
    └── TextConditionerA/
        └── tokenizer/
```

The managed pull uses the `AEmotionStudio/woosh-models` Hugging Face mirror for
Sony Research Woosh v1.0.0 weights and mounts `FacebookAI/roberta-large`
tokenizer files under `checkpoints/TextConditionerA/tokenizer/`. The native
runtime exposes the text-to-SFX distilled DFlow path through
`mere.run sfx generate`.

### `sfx-woosh-flow`

The Woosh original Flow model root is:

```text
.../models/sfx-woosh-flow
└── checkpoints/
    ├── Woosh-Flow/
    ├── Woosh-AE/
    └── TextConditionerA/
        └── tokenizer/
```

The managed pull uses the same mirror and tokenizer mount as
`sfx-woosh-dflow`. The native runtime exposes the original text-to-SFX Flow
checkpoint through `mere.run sfx generate --model sfx-woosh-flow`; it generally
needs more denoise steps than the distilled model.

### Woosh CLAP and V2A

`sfx-woosh-clap` installs `checkpoints/Woosh-CLAP/` plus the mounted
RoBERTa tokenizer for native text/audio scoring through
`mere.run sfx clap score`.

`sfx-woosh-dvflow-8s` and `sfx-woosh-vflow-8s` install the distilled and
original video-to-audio checkpoint stacks from `AEmotionStudio/woosh-models`.
Both include `checkpoints/Woosh-AE/` and `checkpoints/TextConditionerV/`.

`sfx-woosh-synchformer` installs the companion
`mmaudio_synchformer_fp16.safetensors` visual extractor from
`Kijai/MMAudio_safetensors`. `mere.run sfx video generate` uses it when the
input is a raw video file; `.npy` inputs can still provide precomputed
Synchformer `synch_out` features directly.

### `sfx-mmaudio-large-44k-v2`

The native MMAudio install combines pinned public artifacts into one managed
root:

```text
.../models/sfx-mmaudio-large-44k-v2
├── mmaudio_large_44k_v2_fp16.safetensors
├── apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors
├── mmaudio_synchformer_fp16.safetensors
├── mmaudio_vae_44k_fp16.safetensors
├── clip/tokenizer.json
└── bigvgan/
    ├── config.json
    └── bigvgan_generator.pt
```

The MMAudio, CLIP, Synchformer, and VAE safetensors come from the pinned
`Kijai/MMAudio_safetensors` snapshot. CLIP tokenizer/config files are mounted
from Apple's pinned DFN5B CLIP repository. BigVGAN-v2 config and generator
weights are mounted from NVIDIA's pinned 44.1 kHz repository. The Swift runtime
loads the official BigVGAN PyTorch state dictionary with a restricted parser;
it does not execute Python or arbitrary pickle globals.

The `hkchengrex/MMAudio` architecture source is MIT-licensed. The released
MMAudio checkpoints are separately CC-BY-NC-4.0 and therefore non-commercial.
Apple's mounted DFN5B CLIP model is separately restricted to research purposes
under the Apple Machine Learning Research Model License Agreement. NVIDIA's
BigVGAN-v2 source and model are MIT-licensed. The managed install retains the
exact Apple and NVIDIA license files beside those components. Review all model
terms before downloading or using generated output.

### `video-ltx-av`

The unified AV model root is:

```text
.../models/video-ltx-av
```

`mere.run video generate --quality draft --model video-ltx-av` is the faster
video-only draft path. The legacy `--variant` selector remains available for
older scripts using this root.
`MERERUN_VIDEO_LTX_MODEL_ROOT` can still point at this layout explicitly.

### `video-ltx23-av-mlx`

The LTX 2.3 MLX split model root is:

```text
.../models/video-ltx23-av-mlx
```

It pulls the distilled split checkpoint from
`dgrauet/ltx-2.3-mlx`, including `split_model.json`, connector weights,
separate video VAE/audio VAE/vocoder files, and the LTX 2.3 upscalers.
`mere.run model pull video-ltx23-av-mlx` also installs the hidden
`text-encoder-ltx-gemma3-12b-4bit` companion used for Gemma 3 prompt
conditioning. Set `MERERUN_VIDEO_LTX_TEXT_ENCODER_ROOT` only when pointing at an
external `mlx-community/gemma-3-12b-it-4bit` checkout.

The native Swift runtime uses this standalone distilled transformer for the
fast `--quality draft` lane. The default `--output-mode video-only` route
retains the checkpoint's joint AV denoising tokens but does not load or decode
the audio VAE/vocoder, and its MP4 contains no audio stream. It can also emit
synchronized AV with `--output-mode audio-video`, but the canonical final
quality path uses `video-ltx23-full-mlx`.
The Unsloth `LTX-2.3-GGUF` checkpoint family is a separate quantized GGUF lane
and is not loaded by the native MLX video runtime.

### `video-ltx23-full-mlx`

The full LTX 2.3 quality root is:

```text
.../models/video-ltx23-full-mlx
```

It pulls the full/dev transformer, the official rank-384 distilled LoRA,
connector, audio VAE, BWE vocoder, video VAE encoder/decoder, and x2 spatial
upscaler from the same immutable `dgrauet/ltx-2.3-mlx` revision. It does not
duplicate the standalone distilled transformer or pull unrelated x1.5 and
temporal upscalers. Hugging Face cache objects are shared with
`video-ltx23-av-mlx` when both models are installed. The hidden Gemma 3
companion is shared as well.

The same bundle drives all official two-stage final-quality contracts. The
default `--quality final --output-mode video-only` route runs the full video
pipeline without requiring audio in the deliverable. With `--output-mode
audio-video`, unified AV jointly denoises guided video and audio latents in
stage one, then refines both after LoRA fusion. A2Vid encodes source audio and
freezes those audio latents through both stages; the original decoded source
segment—not VAE-decoded audio—is muxed into the MP4.

### `video-ltx23-a2vid-mlx`

This deprecated compatibility ID preserves existing A2Vid installs and scripts:

```text
.../models/video-ltx23-a2vid-mlx
```

Its legacy narrow manifest may omit the vocoder, so it can run final-quality
video-only and source-audio A2Vid but not generated-audio output. Requests for
either the legacy ID or the replacement full ID resolve to an already-installed
compatible root when possible. Use `video-ltx23-full-mlx` for pulls.

### `video-ltx25-distilled-bf16`

The official packed LTX 2.5 BF16 root is:

```text
.../models/video-ltx25-distilled-bf16
```

`mere.run model pull video-ltx25-distilled-bf16 --accept-model-license` pulls
the complete public runtime from
`Sawfwair/LTX-2.5-Distilled-BF16-MLX-Q4-Text` at immutable revision
`cf8a174746cd14796c81ca2b54e035dc32e69bd8`. It stores the upstream BF16
distilled transformer directly in mere.run's native module-key layout, together
with the shared connector, video VAE, audio VAE/BWE vocoder, spatial upsampler,
and duration head from `Lightricks/LTX-2.5@dd53cc2...`. The custom Gemma 4
language tower is MLX affine Q4/group-64; its LTX projection and tokenizer
support remain BF16/raw. The complete selected payload is exactly
53,878,517,792 bytes. The repository is ungated, includes the governing LTX and
Gemma terms and modification notice, and requires no separate component pull,
local quantization, or post-install re-keying.

This model runs natively through Swift and MLX; no Python process or sidecar is
dispatched. Use `--quality final --output-mode audio-video` for synchronized
video and stereo audio.

### `video-ltx25-full-bf16`

The complete official LTX 2.5 root is:

```text
.../models/video-ltx25-full-bf16
```

`mere.run model pull video-ltx25-full-bf16 --accept-model-license` pulls
`Sawfwair/LTX-2.5-Full-BF16-MLX` at immutable revision
`ac74d124f7211fc3cb8b32f418a08d8e71655c8d`. It includes native-layout dev and
distilled BF16 transformers, one shared connector, the official BF16 text
encoder, distilled LoRA, diffusion video VAE, temporal x2 latent upsampler, and
duration head. The complete selected payload is exactly 119,718,579,164 bytes.
The repository is gated and needs no local optimization or source-transformer
copy. This root supports full/dev and HQ pipelines, source-audio A2Vid, DFR,
Retake, HDR/EXR, IC-LoRA reference video, Dub-It, and native text-to-audio.

The official DFR pixel-space spatial upscaler is a separately gated managed
adapter: `ltx25-pixel-spatial-upscaler-x2`. Its repository gate and
`mere.run adapter pull ... --accept-license` acknowledgement are independent
of the main model gate.

See [LTX 2.5 upstream parity](./ltx25-upstream-parity.md) for the exact pinned
code release and pipeline matrix.

### MiniMax-H3 FL2VA and Ref2VA

The native MiniMax-H3 implementation covers both released 33B dense
partitions: FL2VA text/first/last-frame conditioning and Ref2VA ordered
image/video/audio conditioning. It jointly denoises 24-channel video and
32-channel stereo-as-batch audio latents, then decodes 24 fps RGB video and
32 kHz stereo audio into one MP4.

The legacy explicit-pull FL2VA root is the flat
`Sawfwair/MiniMax-H3-FL2VA-MLX-4bit` package pinned at immutable Hub commit
`e1244ad93d60c737c7e0f065a1c9372f3de7caf8`.
Every tensor in that package is derived directly from
`MiniMaxAI/MiniMax-H3@ec19cc6daf5d8add9417c18e86b6b58cc6c55027`; converted or
quantized third-party weights are not inputs. The transformer core is affine
Q4/group-64 with dense precision islands, the exact 50-layer Qwen3-VL
conditioner is affine Q8/group-64, the video VAE is FP16, and the audio VAE has
its released weight normalization folded for the native runtime. The 14-file
managed download is exactly 46,250,104,566 bytes and also carries the tokenizer,
configuration, source manifest, conversion receipt, hashes, `LICENSE`,
`NOTICE`, and `MODIFICATIONS.md`. Before Q4 packing, all 52 fused transformer
QKV matrices are deinterleaved from MiniMax's released per-head row order into
the global Q/K/V slabs consumed by the native runtime. Runtime auto-download is
disabled. This compatibility ID remains installable, but Q4 did not meet the
release visual-quality bar and is no longer recommended.

The maximum-fidelity `video-minimax-h3-fl2va-bf16-mlx` ID now resolves to the
single-root `Sawfwair/MiniMax-H3-FL2VA-MLX-BF16` compact artifact pinned at
immutable Hub commit `4ce4b1d870f7b1b0c75672fd4f2867c1f5df7b5f`. Its active
20.11B-parameter denoising core remains BF16 while the Q8 conditioner, FP16
video VAE, FP32 audio VAE, tokenizer, license, provenance, and cache pack live
beside it. The verified runtime payload is exactly 77,094,088,403 bytes. The old
full transformer overlay remains recoverable from
`pipenetwork/MiniMax-H3-MLX-bf16@1486555759eed9e3037edf29f9e055a0713bab2f`,
but is no longer a managed download source.

The production AdaLN tables in both compact artifacts are generated from the
pinned official projection tensors by `mere.run model optimize` on MLX Metal.
Their source-closure receipt includes exact matched 9- and 21-point real-media
hashes. A streaming MLX 0.32.1 Metal refresh adds the exact LightX2V 9-point
shifts-6/3 table and proves that the seven pre-existing tables retain their
tensor closures. CUDA is still used to reproduce and quantize the large
transformer cores, but CUDA-generated BF16 modulation tables are rejected
because backend reduction order is not bit-identical to Apple Silicon.

`video-minimax-h3-fl2va-8bit-mlx` uses the sibling
`Sawfwair/MiniMax-H3-FL2VA-MLX-8bit` artifact pinned at immutable Hub commit
`86500cb6ebec22c006597e41840b26ef1099fdd7`. Eligible core linears use MLX
affine INT8/group-64; the conditioner, VAEs, tokenizer, cache tables, and
official-source provenance remain the same as compact BF16. Its managed
runtime payload is exactly 58,308,237,969 bytes. Q8 is a disk and
memory option instead of a speed claim. Both compact packages are explicit-pull
only and runtime auto-download remains disabled.

The explicit-pull Ref2VA root is the flat
`Sawfwair/MiniMax-H3-Ref2VA-MLX-8bit` package pinned at immutable Hub commit
`61dc387ef1a7166425cdacd63c2340598dcc364f`. Its 14-file managed download is exactly
70,941,103,245 bytes and includes the complete runtime root, a source-bound
31-point AdaLN cache, source manifest, conversion receipt, hashes, license,
notice, and modification disclosure. Runtime auto-download is disabled.
Eight-bit is the supported Ref2VA floor because lower precision did not meet
the visual quality bar.

The audited release converter accepts only
`minimax_h3_ref2va_int8_convrot.safetensors` from
`Comfy-Org/MiniMax-H3@fd70b39279d1ae6eb214c903f53e1bec3af19a77`, exactly
34,038,894,550 bytes with SHA-256
`9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9`.
It validates each tensor's embedded ConvRot group size, reverses that
regular-Hadamard basis, reproduces MLX affine INT8 group-64 packing, and emits
a hashed receipt. The source uses group 256 for 200 transformer matrices and
group 64 for 50 AdaLN matrices; these source groups are independent of MLX's
output group size. The verified CPU conversion from the script's pinned
PyTorch 2.7.1 toolchain is 36,024,412,656 bytes with SHA-256
`234f22f69f8d40d6ed81cceed8259fa287f3c9417d40fba5274e3a7aa84e18a2`.
It is published as `transformer.safetensors` beside the exact FL conditioner,
VAEs, tokenizer, notices, and a `config.json` whose `partition` is `ref2va`.

`convert_minimax_h3_official_mlx.py` creates the managed FL2VA packages in one
audited pass from the official release. It preserves the active transformer as
BF16 or directly quantizes eligible core linears to affine Q8/group-64 and
always quantizes the conditioner to Q8 once. Before omitting the schedule-only
AdaLN projections, timestep MLP, and reconstructed RoPE tensors, it evaluates
source-bound exact tables for 5, 9, 12, 16, 21, and 31 points at video/audio
shifts 12/3 plus the LightX2V 5- and 9-point shifts 6/3 schedules. The versioned pack
index binds each geometry to its filename, byte count, SHA-256, and official
transformer identity. Custom schedules interpolate from the densest table and
are visibly disclosed as not bit-exact.

The model weights use the MiniMax-H3 Community License, not Apache-2.0. At the
pinned official source revision
`ec19cc6daf5d8add9417c18e86b6b58cc6c55027`, the license excludes use,
distribution, and display in the United States, European Union, United
Kingdom, and Republic of Korea and imposes notice, modification-disclosure,
and safety obligations on downstream distribution. The CLI therefore requires
explicit license acceptance for the managed FL pull, and conversion must be
performed only where the model terms permit it. The upstream FL2VA and Ref2VA
trees are about 144 GB each; the complete upstream repository is roughly
498 GB, so compile success is not artifact or generation proof.

The optional `minimax-h3-turbo-4step` runtime adapter is the EMA checkpoint
`minimax_h3_turbo_4step_ema_ckpt850.safetensors` from
`larryvrh/MiniMax-H3-Turbo-Lora`, pinned at immutable commit
`b604dd5fe25c4c747699f698a1e63f6c46d4a066`. The catalog verifies its exact
779,849,816-byte length and SHA-256
`5a6eeba171cf183020a4ad48774bb2968f29f8168afd6ec17a04987f3528b4ea`.
The adapter card declares Apache-2.0, but using it does not replace or relax
the MiniMax-H3 Community License governing the required BF16 base model.

The managed `minimax-h3-fasth3-vsa-datafree-4step` adapter pins
`vsa-datafree/adapter_model.safetensors` from
`FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA` at immutable revision
`bcf40ca6f457ed66f8badf13514943e390205fca`. The catalog verifies its exact
5,339,117,712-byte length and SHA-256
`42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470`.
The adapter contains 362 LoRA pairs, 82 direct differences, and 50 VSA-H3
compression-gate weights. The runtime consumes the released `fastvideo-lora-v2`
format without producing a fused checkpoint.

The managed `video-minimax-h3-fasth3-vsa-datafree-mlx` package stores a
premerged affine Q8/group-64 student in
`Sawfwair/MiniMax-H3-FastH3-VSA-DataFree-MLX-Q8` at immutable revision
`6068ae3dafafb1e4b2afb29f3109745a16912e07`. The package also includes the Q8
text encoder, both VAEs, tokenizer, Q8 compression gates, and source-bound AdaLN
table. One explicit license-accepting pull installs every asset used by
supported FastH3 inference. The runtime doesn't fetch a second model or adapter
during generation.

FastH3 changes the base transformer's time and AdaLN projections. The
preparation script range-reads the required tensors from
`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` at immutable
revision `b65818d41939b5085451074fe8ca8b799f8d4921`. The script verifies the
transformer index SHA-256, evaluates the four released DMD points with MLX
Metal, and stores the source-bound cache beside the compression gates. The
offline merge tool verifies all 362 LoRA pairs, 82 direct differences, and 50
compression gates. It merges 208 inference linear targets with FP32
accumulation, rounds once to BF16, and encodes the result as MLX affine
Q8/group-64. The runtime loads the premerged transformer and doesn't apply live
LoRA deltas. The MiniMax-H3 Community License continues to govern the base and
student weights.

Another managed H3 adapter, `minimax-h3-lightx2v-4step`, pins
`minimax_h3_fl2v_turbo_4step_v0.1.safetensors` from
`lightx2v/Minimax-h3-Turbo` at immutable commit
`b65e359c0d128b3c5e08e0f5bf2791b794378588`. The catalog verifies its exact
1,383,677,888-byte length and SHA-256
`5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29`.
The runtime consumes its 312 published PEFT pairs directly, applies the
published alpha/rank scale, and projects its separate Q, K, and V deltas into
the base transformer's global slabs without producing an expanded converted
checkpoint. It fuses each scaled delta into the BF16 transformer once during
checkpoint. Standard and QKV projections run as activation-space low-rank
wrappers around dense or stock MLX affine linears; base weights are never fused
or expanded. Its Apache-2.0 adapter
license likewise does not replace the base model's MiniMax-H3 Community
License.

The v1.0 managed adapters `minimax-h3-lightx2v-8step-v1` and
`minimax-h3-lightx2v-4step-v1-768p` pin the non-ComfyUI BF16 checkpoints at
immutable repository revision `e6346777701aa2b64d42ed058cdd71ae00e7cd52`.
Their exact sizes and SHA-256 digests are 1,383,677,768 bytes /
`e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f`
and 1,383,677,808 bytes /
`1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c`.
The runtime binds each filename to the upstream recipe so the 8-step release
uses shifts 12/3 and alpha 8, while the 1344x768 four-step release uses shifts
6/3 and alpha 128.

The `minimax-h3-lightx2v-8step-v1-768p` adapter pins the non-ComfyUI BF16
checkpoint `minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors` at
immutable repository revision `05ef678438e84933c406131b59abbf86919b3aac`.
The catalog verifies its exact 1,383,677,808-byte length and SHA-256
`9b0efe3613b43a84e30febaa43af27432ea9d0711eac7bba904b2556b175f6d4`.
The checkpoint contains 312 BF16 PEFT pairs at rank 128 and declares alpha 8.
mere.run selects eight model evaluations and video/audio shifts 6/3 for its
1344x768 recipe. The pinned compact model roots select the exact source-bound
nine-point AdaLN table for this schedule; full BF16 source roots compute the
same table from their AdaLN weights.

The Ref2VA-specific `minimax-h3-lightx2v-ref2v-4step-v0.1` adapter pins the
non-ComfyUI BF16 checkpoint
`minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` at immutable repository
revision `5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1`. The catalog verifies its
exact 1,383,677,768-byte length and SHA-256
`9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99`.
The checkpoint contains 312 BF16 PEFT pairs at rank 128. mere.run applies the
published alpha 8 and video/audio shifts 12/3, expands the managed INT8 Ref2VA
transformer to resident BF16, and fuses the adapter once before its four
denoise evaluations. The adapter is Apache-2.0; the required base model remains
governed by the MiniMax-H3 Community License.

### `video-wan22-ti2v-5b-mlx`

The native Wan2.2 TI2V-5B model root is:

```text
.../models/video-wan22-ti2v-5b-mlx
```

`mere.run model pull video-wan22-ti2v-5b-mlx` installs a pinned MLX conversion
of `Wan-AI/Wan2.2-TI2V-5B`. The root contains the single 5B transformer, local
UMT5 tokenizer and encoder, and the float32 48-channel Wan2.2 VAE required for
image conditioning. The managed snapshot is pinned by revision; inference does
not fetch tokenizer or model components at runtime.

The core runtime also exposes `Wan2WorldSession` for long-lived world
generation. One session retains the models, prompt cache, and terminal-frame
latent across transitions; callers receive MP4/PNG state artifacts and opaque
state IDs rather than mutable MLX tensors. Camera controls are represented as
XYZ translation and rotation so a DreamX-derived causal camera conditioner can
replace the text-plus-first-frame mode without changing the session
schema.

### `video-scail2-14b-mlx`

The native SCAIL-2 model root is:

```text
.../models/video-scail2-14b-mlx
```

The pinned `Sawfwair/SCAIL-2-14B-MLX` release package contains sharded BF16 transformer
and UMT5 weights, FP16 OpenCLIP weights, the Wan 2.1 VAE, tokenizer,
configuration, MIT license, model card, and a conversion receipt with immutable
source and artifact hashes. The package is approximately 43 GiB.

The mere.run repository contains only the native Swift/MLX runtime. It does not
ship a checkpoint converter or an upstream Python reference implementation.
Install the immutable managed snapshot with:

```bash
mere.run model pull video-scail2-14b-mlx
```

An explicitly prepared compatible root can still be supplied with
`--model-root`.

The optional `scail2-lightx2v-4step` adapter is not part of that model package
or this repository. `mere.run adapter pull` downloads only
`wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors` from
`lightx2v/Wan2.1-Distill-Loras` at immutable revision
`27ae38da91014b947dd39cc3fa78b97cd7b386dd`, verifies 739,472,104 bytes and
SHA-256 `8833bd4fd7c8eabebf0bc8ee5cfaf47f4f310ce116928a02c1adf8941dd4b0f1`,
and stores it outside git under the managed adapter directory. The upstream
model card declares Apache-2.0. The Wan 2.2 adapters are deliberately rejected
for SCAIL-2 because they target the incompatible Wan 2.2 MoE base.

### `video-cosmos3-edge-mlx`

The native Cosmos3-Edge root is the complete official
`nvidia/Cosmos3-Edge` snapshot pinned at revision
`6f58f6b4c91288838e60b6bcb2cc45d997e961de`:

```text
.../models/video-cosmos3-edge-mlx
```

The approximately 9.2 GB snapshot includes the mixed
understanding/generation transformer, Wan VAE, scheduler, generation tokenizer,
reasoner tokenizer/configuration, packed SigLIP2 vision encoder, and multimodal
projector. The Swift/MLX runtime loads these official safetensors directly; it
does not invoke Python, PyTorch, or Diffusers during inference.

```bash
mere.run model pull video-cosmos3-edge-mlx
mere.run guide video-cosmos3
```

The checkpoint is governed by NVIDIA Open Model Development and Use License
1.1 (`OpenMDW-1.1`). Exercising the licensed rights constitutes acceptance;
the public, ungated pull does not require mere.run's acceptance flag.
The managed snapshot retains `LICENSE.md`, and runtime auto-download remains
disabled.

Numerical parity fixtures are generated against NVIDIA's Cosmos framework
commit `ed8287fd7477113f8ac4f6b84290514d55cf0cdc`; the VAE/scheduler reference is
Diffusers v0.39.0 commit `a3608b512ed7248499a44c61d954965ed9bdae4d`.

### `video-dreamx-world-5b-ar-mlx`

The native DreamX-World autoregressive checkpoint root is:

```text
.../models/video-dreamx-world-5b-ar-mlx
```

The public CLI does not convert this local-only checkpoint. Supply a previously
converted `GD-ML/DreamX-World-5B` root at the documented managed path or pass its
directory to `mere.run world serve --model`. The runtime pairs it with
`video-wan22-ti2v-5b-mlx` for tokenizer, text encoder, and VAE resources. The
checkpoint provides learned camera conditioning, block-causal attention,
persistent attention caches, and autoregressive forcing for long-lived local
world sessions. It is never downloaded or converted automatically at runtime.

See [Persistent world runtime](./runtime/world.md) for the server and request
lifecycle.
