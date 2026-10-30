# Architectural decisions

## 1. The CLI is modality-first

The public entry point is organized as `mere.run <modality> <action>`, not by backend or model family. This keeps the surface predictable for users and lets tests assert the public contract at the command-tree layer.

## 2. Public model IDs are canonical and hard-cut

The OSS repo keeps model IDs and command names canonical in normal runtime paths. `scripts/check.sh` rejects retired product vocabulary in primary docs and CLI surfaces so contributors keep examples aligned with the public contract.

## 3. Real-model tests are env-gated

Core correctness needs unit and integration coverage that runs everywhere, but some runtime paths require large local checkpoints or GPU-backed execution. Those tests stay in the tree and skip unless you set an explicit environment variable, such as a `MERERUN_*` model-root path, so contributors can still discover the intended validation path.

## 4. Vendored native runtimes are intentional

`vendor/` holds prebuilt artifacts that the package builds against or bundles: `llama.xcframework`, `magentart.xcframework`, the `ds4` binaries, and the MLX shader bundle `mlx-swift_Cmlx.bundle`. They're checked in so the standalone repo builds without reaching into an internal monorepo payload structure. Treat `vendor/` as an external artifact boundary, not a normal source directory, and record every change in `THIRD_PARTY_NOTICES.md`.

## 5. Compatibility parsing must be explicit

Some model and tokenizer formats come from external ecosystems and evolve independently of mere.run. When a boundary cannot use direct `Decodable`, isolate the compatibility shim close to the ingestion point and document why it remains dynamic instead of letting raw dictionaries spread through the runtime.

## 6. Dynamic model JSON is a typed boundary

External model JSON can have dynamic keys, legacy numeric encodings, or open tool schemas, but raw `Any` dictionaries aren't the normal representation inside `MereRunCore`. Prefer typed `Codable` payloads, dynamic `CodingKey` containers, or small lenient scalar wrappers at the ingestion boundary, then pass domain types through the runtime.

## 7. Agent readiness is an executable gate

Agent orientation files are treated like source code. `scripts/agent_readiness_check.sh` keeps `CODEBASE.md` under 500 words, requires module READMEs for large source directories, and inventories remaining raw dynamic JSON boundary files so new untyped parsing cannot spread without an explicit follow-up decision.
