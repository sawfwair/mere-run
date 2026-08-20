# mere.run documentation

Use this documentation to learn the public `mere.run` package, CLI, and optional
macOS Studio. The pages are organized by task, runtime family, and contributor
workflow so you can find the relevant command or source module directly.

## Run the docs site locally

The repository includes a VitePress site for these docs. Install the site
dependencies, and then start the local development server:

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

The repository also includes a GitHub Pages workflow at
`.github/workflows/docs.yml`. On pull requests it builds the site as a docs-only
CI job, and on `main` it deploys the VitePress output to Pages at
`https://docs.mere.run/`.

## Start here

If you are new to the repo, read these in order:

1. [Getting started](./getting-started.md)
2. [macOS deep links](./macos-deep-links.md), if a local tool should preview or import artifacts in MereRun
3. [Linux quickstart](./linux-quickstart.md), if you are installing the headless CLI on Linux
4. [CLI reference](./cli.md)
5. [Portable workflows](./workflows.md), if you are automating local or remote jobs
6. [Graph Studio](./graph/studio.md), if you want to author Graph v2 workflows visually
7. [Benchmarking](./benchmarking.md)
8. [External evaluation packs](./evaluation-packs.md)
9. [Cookbooks](./cookbooks.md)
10. [Configuration](./configuration.md)
11. [Model sources](./model-sources.md)
12. [Companion plugins](./plugins.md)
13. [Repository tour](./repository-tour.md)

## Choose your path

### I want to use `mere.run`

- [Getting started](./getting-started.md)
- [macOS deep links](./macos-deep-links.md)
- [Linux quickstart](./linux-quickstart.md)
- [CLI reference](./cli.md)
- [Benchmarking](./benchmarking.md)
- [External evaluation packs](./evaluation-packs.md)
- [Cookbooks](./cookbooks.md)
- [Portable workflows](./workflows.md)
- [Graph Studio](./graph/studio.md)
- [Configuration](./configuration.md)
- [Model sources](./model-sources.md)
- [Companion plugins](./plugins.md)

### I want to contribute code

- [Repository tour](./repository-tour.md)
- [Development workflow](./development-workflow.md)
- [Testing guide](./testing.md)
- [Documentation style](./documentation-style.md)
- [Architecture reading map](./architecture.md)
- [CLI and runtime internals](./internals/cli-and-runtime.md)

### I want to understand the runtime families

- [Image runtime](./runtime/image.md)
- [Text runtime](./runtime/text.md)
- [Speech runtime](./runtime/speech.md)
- [Vision runtime](./runtime/vision.md)
- [Music runtime](./runtime/music.md)
- [SFX runtime](./runtime/sfx.md)
- [Video runtime](./runtime/video.md)
- [Persistent world runtime](./runtime/world.md)
- [Model management](./runtime/model-management.md)
- [Local API server](./runtime/api-server.md)

## Documentation map

### Fundamentals

- [Getting started](./getting-started.md): clone, build, first commands, first
  status checks, Linux release artifacts, model pulls, and local setup
- [macOS deep links](./macos-deep-links.md): preview or import completed local
  artifacts from launchers, automations, agents, and other macOS apps
- [Raycast example integration](./raycast.md): one launcher client built on the
  public macOS deep-link surface
- [Linux quickstart](./linux-quickstart.md): Linux package install, first
  commands, release asset verification, and CUDA validation boundaries
- [Cookbooks](./cookbooks.md): `mere.run guide` command topics for practical
  prompting, parameters, examples, and troubleshooting
- [Configuration](./configuration.md): supported environment variables and
  debug toggles
- [Model sources](./model-sources.md): canonical model IDs, Hugging Face sources,
  and local model-store behavior
- [LTX 2.5 upstream parity](./ltx25-upstream-parity.md): pinned upstream
  pipeline matrix, native controls, and hardware-specific boundaries
- [Benchmarking](./benchmarking.md): local quality evals, generated-code
  execution, VLM datasets, API workload, and runtime microbenchmarks
- [External evaluation packs](./evaluation-packs.md): content-addressed private
  packs, adapter comparisons, calibration, gates, and promotion receipts
- [Portable workflows](./workflows.md): typed graphs, immutable job bundles,
  local execution, SSH and relay executors, run artifacts, and remote lifecycle
- [Graph Studio](./graph/studio.md): visual Graph v2 authoring in the browser or
  a cross-platform Tauri desktop app, using the same version-1 workflow contract
- [Companion plugins](./plugins.md): public catalog discovery, safe installation,
  doctor checks, and the typed graph-provider boundary

### Repository guides

- [Repository tour](./repository-tour.md): top-level layout, SwiftPM targets,
  and where each subsystem lives
- [Development workflow](./development-workflow.md): day-to-day edit, build,
  test, Linux packaging, and smoke-test loop
- [Testing guide](./testing.md): what each validation command does and when to
  run it
- [Documentation style](./documentation-style.md): voice, headings,
  accessibility, example data, command examples, and review checks
- [Documentation style audit](./documentation-style-audit.md): line-by-line,
  targeted, and pending review status for every public Markdown file
- [Architecture reading map](./architecture.md): code-reader-oriented entry
  points for each runtime family

### Runtime family guides

- [Image runtime](./runtime/image.md)
- [Text runtime](./runtime/text.md)
- [Speech runtime](./runtime/speech.md)
- [Vision runtime](./runtime/vision.md)
- [Music runtime](./runtime/music.md)
- [SFX runtime](./runtime/sfx.md)
- [Video runtime](./runtime/video.md)
- [Persistent world runtime](./runtime/world.md)
- [Model management](./runtime/model-management.md)
- [Local API server](./runtime/api-server.md)

## Keep command docs synchronized

The generated command inventories on the docs home, Getting started, and CLI
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

- [CLI and runtime internals](./internals/cli-and-runtime.md)
- [Source layout reference](./internals/source-layout.md)

## What this docs set is for

These docs are intentionally split by audience:

- End users can install models and run commands without reading
  source code
- Contributors can understand the package layout and validation
  flow before editing runtime code
- Code readers can follow a command into the correct subsystem without
  searching unrelated modules

Before you open the code, start with the
[repository tour](./repository-tour.md).
