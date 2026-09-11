# Package responsibilities and compatibility

This guide is for contributors who change package dependencies or move runtime
behavior. Use an existing owning target unless a new target provides an
independently testable dependency boundary or serves a named consumer.

`scripts/package-policy.json` records each target's primary role, owner,
purpose, manifest configurations, and allowed direct and transitive dependencies.
It records products separately: an internal target does not require a product.
A model target can have a public product without receiving a second primary role.

Run `./scripts/check-package-policy.sh` to compare the complete target and product
lists from `swift package dump-package` against the registry. Both repository
gates run this check. It covers the Darwin manifest, Linux source linkage, and
Linux prebuilt CUDA linkage. Optional source directories and binary adapters use
explicit applicability conditions. Dependency checks use the union of platform
conditions within each emitted manifest; this conservative closure does not
claim that every dependency links into every Apple platform slice.

The checker rejects missing, duplicate, and stale applicable classifications,
unknown local dependencies, unapproved direct or transitive edges, and production
imports of test targets or test support. Regression fixtures exercise these
failures. Roles come from explicit registry entries, not target-name patterns.

## Change the dependency graph

Before adding an edge, name the consumer and explain why the existing ownership
boundary cannot serve it. Update the affected registry entries deliberately;
do not regenerate an allowlist to make a failing check pass. Keep the narrower
model and portable-operation checks in `check-model-boundaries.swift` intact.

For a structural change, record the before-and-after consumer closure, clean and
incremental build times, and linked binary size. Identify the commits, toolchain,
configuration, retained dependency artifacts, incremental edit, and repetitions.
A target-count reduction is not evidence of a better boundary. Report variance
and explain regressions or incomplete measurements.

## Preserve published imports

The product inventory distinguishes supported SDKs, supported tools, and
implementation exposures awaiting a disposition. The recorded consumers are
in-repository evidence; they are not a complete audit of external package users.
The existing implementation-exposure inventory is a baseline, not permission
to add another implementation-only product. A new public surface needs an
explicit supported consumer and a compatibility commitment.

Every re-export has an owner, purpose, known consumers, migration, and removal
condition. The guard compares those records with the source's `@_exported import`
declarations. Move callers to explicit imports where appropriate, but retain
published compatibility until the stated removal condition is met. Do not
remove an import surface to meet an arbitrary target count or deadline.
