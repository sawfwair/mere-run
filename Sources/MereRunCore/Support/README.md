# Core Support

Shared support code for model sources, archives, signing, debug, and lenient
JSON primitives.

- `MereRunModelSourceConfiguration.swift`: model-source environment handling.
- `PretrainedModelLoader.swift`: download and cache support.
- `ArchiveIntegrity.swift` and `AWSV4Signer.swift`: integrity/auth helpers.
- `LenientJSON.swift`: typed scalar wrappers for compatibility boundaries.

Do not add product-specific policy here unless it is shared by multiple runtime
families.
