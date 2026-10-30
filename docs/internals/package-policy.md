# Package responsibilities and compatibility

Use this page when you change package dependencies or move runtime behavior
between targets. Prefer an existing owning target. Add a new target only when
it gives an independently testable dependency boundary or serves a named
consumer.

## Ownership

`scripts/package-policy.json` is the registry of record. For each target, it
records the primary role, owner, purpose, manifest configurations, and allowed
direct and transitive dependencies. Products are recorded separately: an
internal target doesn't need a product, and a model target can have a public
product without taking a second primary role. Roles come from explicit registry
entries, not from target-name patterns.

`./scripts/check-package-policy.sh` compares the complete target and product
lists from `swift package dump-package` against the registry. Both repository
gates run it. It covers the Darwin manifest, Linux source linkage, and Linux
prebuilt CUDA linkage. Optional source directories and binary adapters use
explicit applicability conditions. Dependency checks use the union of platform
conditions within each emitted manifest; this conservative closure doesn't
claim that every dependency links into every Apple platform slice.

The checker rejects:

- Missing, duplicate, and stale applicable classifications
- Unknown local dependencies
- Unapproved direct or transitive edges
- Production imports of test targets or test support

Regression fixtures exercise each of these failures. The narrower model and
portable-operation checks live in `scripts/check-model-boundaries.swift`; keep
them intact.

## Change the dependency graph

Before you add an edge, name the consumer and explain why the existing
ownership boundary can't serve it. Update the affected registry entries
deliberately; don't regenerate an allowlist to make a failing check pass.

For a structural change, record the before-and-after consumer closure, clean
and incremental build times, and linked binary size. Identify the commits,
toolchain, configuration, retained dependency artifacts, incremental edit, and
repetitions. Report variance, and explain regressions or incomplete
measurements. A lower target count isn't evidence of a better boundary.

## Preserve published imports

The product inventory distinguishes supported SDKs, supported tools, and
implementation exposures that await a disposition. Its recorded consumers are
in-repository evidence, not a complete audit of external package users. The
existing implementation-exposure inventory is a baseline, not permission to add
another implementation-only product. A new public surface needs an explicit
supported consumer and a compatibility commitment.

Every re-export has an owner, purpose, known consumers, migration, and removal
condition. The guard compares those records with the source's
`@_exported import` declarations. Move callers to explicit imports where it
makes sense, but keep published compatibility until the stated removal
condition is met. Don't remove an import surface to reach an arbitrary target
count or deadline.

## Validate a change

```bash
./scripts/check-package-policy.sh
bash scripts/check-model-boundaries.sh
./scripts/check.sh
```
