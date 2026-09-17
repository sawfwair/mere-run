# MereRunRelayKit

The portable client half of Relay execution: executor profiles and references,
OAuth device-grant authentication, the relay graph-job and fleet HTTP client,
and the serialized workflow contract types those APIs speak (`WorkflowValue`,
job/asset manifests, `GraphRun*` records and events).

This target exists so user-interface shells — the macOS Studio and the iOS app
— can watch, fetch, and manage relay runs without importing a model runtime or
the CLI. It depends on Foundation and swift-crypto only, builds for macOS,
iOS, and Linux, and never owns platform paths or process spawning:

- Storage locations (executor profiles, token files) are supplied by the
  caller. The CLI passes its application-support base; app clients pass their
  sandbox container.
- Errors are `RelayClientError`. The CLI maps them onto its argument-parsing
  error at the command boundary so messages and exit codes are unchanged.
- Bundle materialization lives here. `WorkflowBundleMaterializer` in
  `WorkflowBundleMaterialization.swift` builds the graph, inputs, asset
  manifest, and job manifest, so `MereRunCLI`
  (`Sources/MereRunCLI/Commands/GraphCommand.swift`) and the iOS Studio
  (`apps/ios/MereRunStudio/App/RelayStore.swift`) produce byte-identical
  bundles and submit through the same create/upload/commit path. Each caller
  supplies a `WorkflowMaterializationEnvironment`: the CLI passes its managed
  model catalog, per-command default models, and discovered plugin providers;
  clients without a local runtime use `.portable` and pin models in the graph.
  The CLI additionally owns the local run-directory record. See
  `docs/ios-studio.md`.

Wire shapes here must stay in lockstep with the published schemas in
`docs/public/schemas/` (graph-event-v1, graph-run-v1, worker-probe-v1,
job-bundle-v1, asset-manifest-v1) and with the relay and node repositories'
shared contract fixtures. Do not change field names or add required fields
without a coordinated contract revision.

Tests live in `Tests/MereRunRelayKitTests`.
