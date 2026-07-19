# mere.run Documentation

This documentation set is organized like a practical manual for the public
`mere.run` package, CLI, and optional macOS studio. The goal is to make it easy to learn the tool,
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
CI job, and on `main` it deploys the VitePress output to Pages at
`https://docs.mere.run/`.

## Start here

If you are new to the repo, read these in order:

1. [Getting Started](./getting-started.md)
2. [Linux QuickStart](./linux-quickstart.md), if you are installing the headless CLI on Linux
3. [CLI Reference](./cli.md)
4. [Portable Workflows](./workflows.md), if you are automating local or remote jobs
5. [Benchmarking](./benchmarking.md)
6. [Cookbooks](./cookbooks.md)
7. [Configuration](./configuration.md)
8. [Model Sources](./model-sources.md)
9. [Companion Plugins](./plugins.md)
10. [Repository Tour](./repository-tour.md)

## Choose your path

### I want to use `mere.run`

- [Getting Started](./getting-started.md)
- [Linux QuickStart](./linux-quickstart.md)
- [CLI Reference](./cli.md)
- [Benchmarking](./benchmarking.md)
- [Cookbooks](./cookbooks.md)
- [Portable Workflows](./workflows.md)
- [Configuration](./configuration.md)
- [Model Sources](./model-sources.md)
- [Companion Plugins](./plugins.md)

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
- [SFX Runtime](./runtime/sfx.md)
- [Video Runtime](./runtime/video.md)
- [Persistent World Runtime](./runtime/world.md)
- [Model Management](./runtime/model-management.md)
- [Local API Server](./runtime/api-server.md)

## Documentation map

### Fundamentals

- [Getting Started](./getting-started.md): clone, build, first commands, first
  status checks, Linux release artifacts, model pulls, and local setup
- [Linux QuickStart](./linux-quickstart.md): Linux package install, first
  commands, release asset verification, and CUDA validation boundaries
- [Cookbooks](./cookbooks.md): `mere.run guide` command topics for practical
  prompting, parameters, examples, and troubleshooting
- [Configuration](./configuration.md): supported environment variables and
  debug toggles
- [Model Sources](./model-sources.md): canonical model IDs, Hugging Face sources,
  and local model-store behavior
- [Benchmarking](./benchmarking.md): local quality evals, generated-code
  execution, VLM datasets, API workload, and runtime microbenchmarks
- [Portable Workflows](./workflows.md): typed graphs, immutable job bundles,
  local execution, SSH and relay executors, run artifacts, and remote lifecycle
- [Companion Plugins](./plugins.md): public catalog discovery, safe installation,
  doctor checks, and the typed graph-provider boundary

### Repository guides

- [Repository Tour](./repository-tour.md): top-level layout, SwiftPM targets,
  and where each subsystem lives
- [Development Workflow](./development-workflow.md): day-to-day edit, build,
  test, Linux packaging, and smoke-test loop
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
- [SFX Runtime](./runtime/sfx.md)
- [Video Runtime](./runtime/video.md)
- [Persistent World Runtime](./runtime/world.md)
- [Model Management](./runtime/model-management.md)
- [Local API Server](./runtime/api-server.md)

## Keep command docs synchronized

The generated command inventories on the docs home, Getting Started, and CLI
reference come directly from `MereRunCLI.configuration`. Every top-level command
also has an explicit owner in `.vitepress/command-pages.tsv`.

After adding, renaming, removing, or redescribing a command, run:

```bash
./scripts/update-docs-command-reference.sh
```

`DocumentationContractTests` fails the normal `swift test` and
`./scripts/check.sh` gates when:

- a generated command inventory differs from the CLI configuration
- a top-level command lacks a canonical docs owner or sidebar entry
- a runtime guide is absent from the sidebar
- a command invocation in Markdown names a subcommand that no longer exists

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
