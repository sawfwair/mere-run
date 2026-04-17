# Migration from the pre-rename CLI

This repo made a hard cut to a modality-first public surface. There are no runtime aliases for the older command tree or older managed model IDs.

## One-time migration steps

1. Update your scripts and docs to call `swift run mere.run ...`
   The old flat executable and command tree are no longer available.
2. Migrate your local model store once:

```bash
./scripts/migrate_model_store.sh --source-models-root /path/to/legacy/models
```

## Store migration behavior

`./scripts/migrate_model_store.sh` does three things:

1. renames top-level installed model directories to the canonical OSS names
2. renames legacy manifest files to `mererun_model.json` and rewrites them to the canonical model IDs and image-family names
3. renames nested Music Acestep subdirectories so `mere.run music generate` resolves the new default layout cleanly

You can preview the migration first:

```bash
./scripts/migrate_model_store.sh --source-models-root /path/to/legacy/models --dry-run
```
