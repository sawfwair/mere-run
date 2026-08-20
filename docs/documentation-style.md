# Documentation style

Use this guide for every public page, example, command description, and pull
request that changes documentation. The goal is consistent documentation that
helps readers complete a task without needing repository context.

## Write for the reader

- Start with the outcome or decision the reader needs.
- Address the reader as "you" when giving instructions.
- Use active voice and present tense.
- Keep sentences and paragraphs focused on one idea.
- Define unfamiliar terms before using abbreviations.
- Use the same term for the same concept throughout a page.
- Separate verified behavior, limitations, and planned work.

Prefer direct instructions:

> Run `mere.run model capabilities` before you download a model.

Avoid indirect wording:

> The model capabilities command can be run by users before a model is
> downloaded.

## Use sentence case

Use sentence case for page titles, headings, table headings, and link text.
Capitalize product names, model names, and other proper nouns as their owners
write them.

- Use **Getting started**, not **Getting Started**.
- Use **Local API server**, not **Local API Server**.
- Keep **Graph Studio**, **Open WebUI**, and **Apple Silicon** capitalized.

## Make links and structure accessible

- Use link text that describes the destination. Avoid "here," "this link," and
  raw URLs when a descriptive label works.
- Do not rely on position, color, or visual styling to convey meaning.
- Give every table a header row and keep list items grammatically parallel.
- Introduce code blocks and tables before they appear.
- Use ordered lists for sequences and unordered lists for collections.
- Add concise alt text to meaningful images. Use empty alt text only for
  decorative images.

## Use unmistakable example data

Public documentation must use fictional people, organizations, workspaces,
hostnames, and account identifiers unless the text identifies a real public
project or source.

- Use reserved email domains such as `dana@example.com`.
- Use reserved DNS names under `example.com`, `example.net`, or `example.org`.
- Use fictional identifiers such as `greenhouse-ops`, `project-alpha`, and
  `deployment-123`.
- Use the documentation IP ranges `192.0.2.0/24`, `198.51.100.0/24`, and
  `203.0.113.0/24`.
- Never copy customer, contributor, internal, or proprietary data into a public
  example.

Project-owned addresses under `mere.run` and accurate third-party attribution
in `THIRD_PARTY_NOTICES.md` are not example data.

Run the example-data check before you open a pull request:

```bash
bash ./scripts/check-docs-examples.sh
```

## Describe versions and time precisely

Prefer a version, date, or named state over words such as "latest" and
"current" when the statement can become stale.

- Write "The v0.41.0 package includes ..." for a release-specific claim.
- Write "As of 2026-08-20, the validation covers ..." for dated evidence.
- Use "current working directory" when it is the technical name of a process
  property.

Do not rewrite a historical benchmark receipt to match a newer runtime. Add a
new dated result and preserve the original hashes, environment, and limitations.

## Document commands completely

For each command workflow:

1. State what the command does and whether it changes local state.
2. List prerequisites before the command.
3. Show a copyable command with realistic reserved values.
4. Describe the expected output or artifact.
5. Explain important failure modes and safety boundaries.

Keep machine-readable output on stdout and diagnostics on stderr when you
describe or change CLI behavior.

## Review documentation changes

Before you open a pull request:

1. Read the changed pages as a new user, not as the implementation author.
2. Check headings, links, lists, tables, code blocks, and example data.
3. Confirm that commands and model IDs match the source.
4. Regenerate command documentation when the command tree changes.
5. Build the documentation site.

```bash
bash ./scripts/check-docs-examples.sh
pnpm docs:build
```

For a command-tree change, also run:

```bash
./scripts/update-docs-command-reference.sh
./scripts/check.sh
```
