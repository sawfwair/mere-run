# iOS Studio

The iOS app (`apps/ios/`) combines the portable relay client established by hosted
Graph Studio with a constrained on-device runtime. It can submit and watch
fleet work, fetch verified artifacts, connect directly to a machine, or run
supported image and chat models on a physical iPhone. `apps/ios/README.md` covers
the build process.

## Choose an execution lane

1. **Relay lane (primary).** Use a paired fleet for the full capability set. The
   phone uses the same authenticated HTTPS and JSON job API as the CLI's
   relay executor, gets its node catalog from the fleet through relay (the
   hosted-Studio pattern. It does not use local discovery, which requires process
   spawning that iOS forbids), and reads the same `graph-event-v1` /
   `graph-run-v1` documents published in `docs/public/schemas/`.
2. **On-device lane (experimental).** `MereRunCore` builds for iOS. The app's
   image and chat modes offer "This iPhone" alongside the fleet and install
   managed models into the app sandbox. These paths require a physical device;
   a simulator build is portability evidence, not inference proof. Sizing:
   chat selection is RAM-aware: 4-5 GB devices prefer the 1.2B QAD MLX model,
   while 6 GB-and-up devices retain the faster standard 2.6B MLX default. The
   2.6B QAD model appears as an 8 GB-and-up quality-recovery option. For example,
   the Bonsai image binary (4B, 1-bit, ~3.4 GB) runs through the same FLUX.2
   Klein pipeline that already has a memory-constrained iOS generator in
   `MereRunCore` (`Flux2KleinGeneratoriOS`, ~2 GB peak via sequential
   loading), and `text-chat-bonsai-27b-1bit` (~5.1 GB, Apache-2.0,
   vision-capable) targets 12 GB+ devices with the increased-memory
   entitlement. The app links the broad `MereRunCore` target; a
   narrower mobile runtime remains a build-time and binary-size optimization.
   CI builds only the selected simulator architecture so test coverage does not
   accidentally turn into a universal two-architecture product-size proxy.

## Components

- `Sources/MereRunRelayKit` — the portable client half of relay execution,
  extracted from `MereRunCLI`: executor profiles and references, OAuth
  device-grant authentication, the graph-job/fleet HTTP client with verified
  artifact fetch, the workflow wire types, and SSE event-text normalization.
  It depends only on Foundation and swift-crypto, builds and tests on Linux,
  and preserves CLI behavior.
- `apps/ios/` — the SwiftUI client: PKCE and device-code relay sign-in, direct
  machine pairing, fleet and run views, polled events/progress, cancel/retry,
  verified artifact fetch and sharing, Live Activities, and on-device model
  management, image generation, and chat. XcodeGen owns the project, and CI
  regenerates and Release-builds it for the simulator.

## Phasing

- **Phase A — run inbox and prompt-first submission (landed).** Pair,
  watch, fetch, cancel/retry, fleet, and Create: the graph documents, node
  registry, validator, and bundle materializer now live in
  `MereRunRelayKit`, so the phone materializes byte-identical bundles and
  submits through the same create/upload/commit path as the CLI. The Create
  surface generates its forms from the shared node catalog (image, video,
  music, SFX, and speech synthesis), reports per-kind fleet availability
  from the worker probe, and pins models from the fleet's installed list.
  Provider-qualified nodes stay excluded on iOS, and modes whose required
  inputs are assets wait for the photo/file picker step.
- **Phase B — chat (landed).** The app routes chat to the selected transport:
  on-device
  small models when present, a machine reached directly when paired (the
  `mere.run relay serve` direct lane — the app's "Connect to a machine"
  pairing serves the full job vocabulary over a LAN or tailnet with no
  cloud hop, superseding the earlier `api serve`-over-LAN idea), relay jobs
  otherwise. Token streaming
  *through relay* is an additive event-type revision, not an architecture
  change: the node→relay leg is a persistent WebSocket, relay serves
  `/events` SSE-framed, and the event `type` is an open string in the
  implementation — but `graph-event-v1` is a closed schema, so a
  `node_output_delta` event needs a coordinated contract revision with the
  relay and node repositories' shared fixtures.
- **Phase C — on-device models (experimental).** The managed Bonsai and Liquid
  lane above is gated by device RAM and available storage. Model payloads use
  one fixed, system-owned background session, so the active file can continue
  while the app is suspended. The app durably records the requested install,
  rejoins matching system tasks after relaunch, stages completed payloads out
  of URLSession's temporary directory, and then resumes the pinned
  snapshot and final validation pipeline. A user-initiated download
  defaults to unmetered, unconstrained Wi-Fi. Users can explicitly allow
  cellular, expensive, and constrained networks before starting a download.
  The app records that choice with the pending install so a relaunch cannot
  silently change the transfer policy.

## On-device lifecycle

- Chat warmup remains immediate, but resident chat or image state is released
  after two idle minutes, when the selected chat model or inference lane
  changes, when the app enters the background, and on an iOS memory warning.
- A lifecycle release does not interrupt active inference. The app defers it until
  the active image or chat request finishes, and the engine rejects another
  request while release is in progress.
- When you remove an installed model, the app unloads all runtime state before deleting its
  install and reclaiming only cache payloads not referenced by another model.
- iOS might relaunch the app to deliver background-session events. The app
  reconnects that session through its application delegate and resumes every
  durable pending install. Explicit user force-quit behavior remains controlled
  by iOS and is not treated as a supported continuation path.
- Each job has one owner for Live Activity polling. Transient request failures use
  bounded exponential backoff; persistent failures end the activity instead of
  leaving an immortal poller behind. Unpairing cancels all outstanding pollers.
- Refreshing a run's artifacts stages and validates the complete replacement result
  beside the existing result, then atomically replaces it. A failed refresh
  leaves the last verified local copy intact.
- Privacy copy follows the selected execution lane: on-device work stays on the
  phone, a direct-machine lane names that machine connection, and a hosted relay
  work describes the authenticated relay/fleet path rather than claiming every
  remote request is local-only.

## Follow-up work

- Keychain-backed credential storage and Authorization Code + PKCE sign-in
  (done: `ASWebAuthenticationSession` over the broker's advertised
  `authorization_endpoint`, universal-link callback on `mere.world`,
  tokens in the Keychain with file→Keychain migration; the device grant
  remains as the browserless fallback).
- Artifact downloads stream to disk with digest verification (done);
  incremental in-flight hashing remains planned.
- Physical-device interruption coverage still needs a release-signed run that
  backgrounds the app during a real multi-gigabyte model pull, verifies
  relaunch reattachment, and observes runtime memory returning after eviction.
- Debug uses the associated-domain developer alternate mode; Release uses the
  production AASA CDN. The app and widget versions must remain identical.
- Live Activities ship with app-driven updates; completion pushes and
  push-updated activities need APNs support in the relay repository.

## Release footprint

A clean arm64 release simulator build on 2026-08-19 measured 75.4 MiB
uncompressed, including a 68.3 MiB main executable and a 4.3 MiB MLX Metal
library. The unused llama.cpp binary dependency is macOS-only. This is not an
App Store download-size measurement, but it confirms that the broad
`MereRunCore` closure is the main compile-time and binary-size cost. CI
rejects a non-arm64-only simulator executable and caps that executable at 80
MiB. Extracting the image/chat/model storage surface into a narrower mobile
runtime remains a separate architectural optimization; a facade that still
depends on `MereRunCore` would not reduce the dependency closure.

## Boundaries

The wire shapes in `MereRunRelayKit` are the published contract. If the iOS app
needs different shapes, revise the cross-repository contract before you add the
app feature. This requirement applies to fields, event types, and similar
changes. The app does not use a private schema.
