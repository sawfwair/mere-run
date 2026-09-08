# MereRunModelKit

Model identities, manifests, configured storage paths, registered locations,
artifact fingerprints, and installed-only lookup without inference dependencies.
The target imports Foundation and Crypto; it does not import Core, MLX, Hub,
ArgumentParser, or runtime families.

`InstalledModelResolver` owns candidate ordering, manifest identity checks,
usage-term acknowledgement, and explicit fallback selection. Callers supply an
`InstalledModelDescriptor` from their catalog and a required runtime validator.
A directory's metadata alone never establishes that its checkpoint is runnable.

`MereRunCore.ModelResolver` delegates lookup here and supplies existing catalog
facts and family validators. Core re-exports these types for source compatibility.
Runtime-dependent manifest templates, catalog assembly, downloads, and tensor
loading remain in Core.
