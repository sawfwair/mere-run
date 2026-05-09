# MereRunApp

Optional SwiftUI studio wrapper around the public `mere.run` CLI.

- `StudioTypes.swift`: user-facing mode, draft, and request types.
- `CommandCatalog.swift`: mode-to-command templates.
- `MereRunController.swift`: child-process launching and log capture.
- `StudioLibraryStore.swift`: local library persistence.

Do not duplicate runtime logic here. The app should translate UI state into CLI
arguments and let the public executable remain the behavioral source of truth.
