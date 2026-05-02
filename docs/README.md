# mere.run Documentation

This documentation set is organized like a practical manual for the public
`mere.run` package and CLI. The goal is to make it easy to learn the tool,
understand the repository, and navigate the runtime code without guessing where
things live.

## Run the docs site locally

The repo ships with a VitePress site for these docs:

```bash
brew install node pnpm
```

```bash
pnpm install
pnpm docs:dev
```

Build the static site with:

```bash
pnpm docs:build
```

The repo also includes a dedicated GitHub Pages workflow at
`.github/workflows/docs.yml`. On pull requests it builds the site as a docs-only
CI job, and on `main` it deploys the VitePress output to Pages.

## Start here

If you are new to the repo, read these in order:

1. [Getting Started](./getting-started.md)
2. [CLI Reference](./cli.md)
3. [Configuration](./configuration.md)
4. [Model Sources](./model-sources.md)
5. [Repository Tour](./repository-tour.md)

## Choose your path

### I want to use `mere.run`

- [Getting Started](./getting-started.md)
- [CLI Reference](./cli.md)
- [Configuration](./configuration.md)
- [Model Sources](./model-sources.md)

### I want to contribute code

- [Repository Tour](./repository-tour.md)
- [Development Workflow](./development-workflow.md)
- [Testing Guide](./testing.md)
- [Architecture Reading Map](./architecture.md)
- [CLI and Runtime Internals](./internals/cli-and-runtime.md)

### I want to understand the runtime families

- [Image Runtime](./runtime/image.md)
- [Text Runtime](./runtime/text.md)
- [Speech Runtime](./runtime/speech.md)
- [Vision Runtime](./runtime/vision.md)
- [Music Runtime](./runtime/music.md)
- [Video Runtime](./runtime/video.md)
- [Model Management](./runtime/model-management.md)
- [Local API Server](./runtime/api-server.md)

## Documentation map

### Fundamentals

- [Getting Started](./getting-started.md): clone, build, first commands, first
  model pulls, and local setup
- [Configuration](./configuration.md): supported environment variables and
  debug toggles
- [Model Sources](./model-sources.md): canonical model IDs, archives, and
  explicit download configuration

### Repository guides

- [Repository Tour](./repository-tour.md): top-level layout, SwiftPM targets,
  and where each subsystem lives
- [Development Workflow](./development-workflow.md): day-to-day edit, build,
  test, and smoke-test loop
- [Testing Guide](./testing.md): what each validation command does and when to
  run it
- [Architecture Reading Map](./architecture.md): code-reader-oriented entry
  points for each runtime family

### Runtime family guides

- [Image Runtime](./runtime/image.md)
- [Text Runtime](./runtime/text.md)
- [Speech Runtime](./runtime/speech.md)
- [Vision Runtime](./runtime/vision.md)
- [Music Runtime](./runtime/music.md)
- [Video Runtime](./runtime/video.md)
- [Model Management](./runtime/model-management.md)
- [Local API Server](./runtime/api-server.md)

### Internal implementation guides

- [CLI and Runtime Internals](./internals/cli-and-runtime.md)
- [Source Layout Reference](./internals/source-layout.md)

## What this docs set is for

These docs are intentionally split by audience:

- end users should be able to install models and run commands without reading
  source code
- contributors should be able to understand the package layout and validation
  flow before editing runtime code
- code readers should be able to follow a command into the correct subsystem in
  a few clicks

If you only read one page before opening the code, read
[Repository Tour](./repository-tour.md).
