# Architectural Decisions

## 1. The CLI is modality-first

The public entrypoint is organized as `mere.run <modality> <action>` rather than by backend or model family. This keeps the surface predictable for users and lets tests assert the public contract at the command-tree layer.

## 2. Public model IDs are canonical and hard-cut

The OSS repo keeps model IDs and command names canonical in normal runtime paths. `scripts/check.sh` rejects retired product vocabulary in primary docs and CLI surfaces so contributors keep examples aligned with the public contract.

## 3. Real-model tests are env-gated

Core correctness needs unit and integration coverage that run everywhere, but some runtime paths require large local checkpoints or GPU-backed execution. Those tests stay in the tree and remain skippable through explicit environment variables so contributors can still discover the intended validation path.

## 4. Vendored native runtime is intentional

`vendor/llama.xcframework` is checked into the package layout so the standalone repo can build without reaching back into an internal monorepo payload structure. Treat that directory as an external artifact boundary, not a normal source directory.

## 5. Compatibility parsing must be explicit

Some model and tokenizer formats come from external ecosystems and evolve independently of mere.run. When a boundary cannot use direct `Decodable`, isolate the compatibility shim close to the ingestion point and document why it remains dynamic instead of letting raw dictionaries spread through the runtime.

## 6. Dynamic model JSON is a typed boundary

External model JSON can have dynamic keys, legacy numeric encodings, or open tool schemas, but raw `Any` dictionaries should not be the normal representation inside `MereRunCore`. Prefer typed `Codable` payloads, dynamic `CodingKey` containers, or small lenient scalar wrappers at the ingestion boundary, then pass domain types through the runtime.
