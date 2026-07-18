# Model Storage

## Purpose

Inspect physical model storage without double-counting shared payloads, then
preview or remove cache data that no installed or legacy MereRun model links use.

## Inspect

```bash
mere.run model storage
mere.run model storage --json
```

The top section reports physical bytes for the application-support directory,
Hub payload store, model links/local files, and other app data. Per-model
`referenced` values describe what each model can read and must not be added
together. `reclaimable on removal` accounts for sharing.

## Clean Up

```bash
# Read-only dry run
mere.run model gc
mere.run model gc --json

# Recompute the plan under the storage lock, then delete it
mere.run model gc --force
```

Garbage collection preserves payloads referenced by the configured model store
or legacy links under MereRun application support. It also coordinates with
managed pulls, rechecks the ownership graph before deletion, and protects newly
created unreferenced snapshots for one hour. Do not run cleanup while an
unmanaged process is directly reading files from the Hub cache.

## Layout Compatibility

Existing `hub/models/<org>/<repo>` payloads remain valid. New pulls use
revision-addressed `hub/snapshots/...` directories and ETag-addressed hard-linked
blobs. A later pull can adopt matching legacy files without downloading or
copying their payload bytes; cleanup removes an old layout only after its last
link disappears.

## Related Commands

```bash
mere.run model list
mere.run model remove <id>
mere.run model remove <id> --keep-cache
```

`model remove` now deletes unshared backing payloads by default and reports the
bytes actually reclaimed. Use `--keep-cache` when you intentionally want a fast
future reinstall.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCore/ModelStorageManager.swift
- https://github.com/sawfwair/mere-run/blob/main/docs/runtime/model-management.md
