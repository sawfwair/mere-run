# Local API Server

This page covers `mere.run api serve`, the local API surface exposed by the package.

## Public surface

- `mere.run api serve`

## What it is for

The API server lets you expose supported local engines through a local process
instead of shelling out to the CLI for every request. It is useful for:

- local automation
- editor tooling
- simple local integrations
- experimenting with the runtime through HTTP

It is not a hosted-service or relay layer. This repo keeps the server local and
package-scoped.

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/APIServeCommand.swift`

### Supporting stack

- `Sources/MereRunCLI/Support/`
- `Hummingbird` package dependency declared in `Package.swift`

## Example

```bash
swift run mere.run api serve --engine text-chat-gemma4
```

## Design notes

- the API server follows the same model-resolution and model-store rules as the
  rest of the CLI
- it is intentionally local-first
- it should not reintroduce relay, billing, or hosted-infrastructure concerns

If you are working on this area, read [CLI and Runtime Internals](../internals/cli-and-runtime.md) after the command source.
