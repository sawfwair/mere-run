# MereRunCore

Shared model resolution, manifests, generation protocols, and native runtime
families.

- `Generation.swift`: image and chat request/response contracts.
- `ManagedModel*.swift`: public managed-model catalog and install metadata.
- `MereRunModel*.swift`: manifests, paths, and validation.
- Runtime family directories own model-specific loading, inference, and decode
  paths.

Keep external config and tokenizer data typed at the boundary. Do not let raw
dynamic JSON move deeper than the compatibility shim that ingests it.
