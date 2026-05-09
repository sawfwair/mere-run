# Core Support

Shared support code for model sources, Hugging Face snapshots, debug, and
lenient JSON primitives.

- `PretrainedModelLoader.swift`: local path, managed store, and Hub snapshot
  resolution.
- `LenientJSON.swift`: typed scalar wrappers for compatibility boundaries.

Do not add product-specific policy here unless it is shared by multiple runtime
families.
