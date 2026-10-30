# iOS Studio

The iOS app in `apps/ios/` is a relay client with a constrained on-device
runtime. It can submit and watch fleet work, fetch verified artifacts, connect
directly to a machine, and run supported image and chat models on a physical
iPhone. For build steps, see `apps/ios/README.md`.

## Choose an execution lane

The app offers two ways to run work.

### Relay lane

The relay lane is the primary lane and offers the full capability set through
a paired fleet. The phone uses the same authenticated HTTPS and JSON job API as
the CLI's relay executor. It reads the same `graph-event-v1` and `graph-run-v1`
documents published in `docs/public/schemas/`.

The phone gets its node catalog from the fleet through the relay. It doesn't
use local discovery, because local discovery spawns processes, which iOS
doesn't allow.

### On-device lane (experimental)

`MereRunCore` builds for iOS. The image and chat modes offer **This iPhone**
alongside the fleet and install managed models into the app sandbox. These
paths require a physical device; a simulator build is portability evidence,
not inference proof.

The app offers these on-device models, filtered by device memory:

| Mode | Model ID | Minimum device memory |
| --- | --- | --- |
| Image | `image-klein-nano` | None |
| Image | `image-bonsai-binary` | None |
| Image | `image-bonsai-ternary` | None |
| Chat | `text-chat-lfm25-1.2b-qad-4bit` | 4 GB |
| Chat | `text-chat-lfm25-2.6b-4bit` | 6 GB |
| Chat | `text-chat-lfm25-2.6b-qad-4bit` | 8 GB |
| Chat | `text-chat-bonsai-27b-1bit` | 12 GB |

Chat selection is RAM-aware. Devices with 6 GB or more default to the faster
2.6B model, and devices with 4 to 5 GB default to the 1.2B QAD model. The 2.6B
QAD model is an optional quality-recovery choice for 8 GB devices and up. The
vision-capable Bonsai 27B chat model (about 5.1 GB, Apache-2.0) targets 12 GB
devices and uses the increased-memory-limit entitlement.

The Bonsai image models (about 3.4 GB for the binary checkpoint) run through
the FLUX.2 Klein pipeline, which has a memory-constrained iOS generator in
`MereRunCore` (`Flux2KleinGeneratoriOS`, about 2 GB peak through sequential
loading).

The app links the broad `MereRunCore` target. A narrower mobile runtime would
reduce build time and binary size; see [Release footprint](#release-footprint).
D1, Laya, Clef, and LightOnOCR inference code is excluded from iOS builds, but
their shared catalog metadata remains available. CI builds only the selected
simulator architecture, so test coverage doesn't turn into a universal
two-architecture product-size proxy.

## Components

- `Sources/MereRunRelayKit`: the portable client half of relay execution,
  extracted from `MereRunCLI`. It holds executor profiles and references, OAuth
  device-grant authentication, the graph-job and fleet HTTP client with
  verified artifact fetch, the workflow wire types, and SSE event-text
  normalization. It depends only on Foundation and swift-crypto, builds and
  tests on Linux, and preserves CLI behavior.
- `apps/ios/`: the SwiftUI client. It provides PKCE and device-code relay
  sign-in, direct machine pairing, fleet and run views, polled events and
  progress, cancel and retry, verified artifact fetch and sharing, Live
  Activities, and on-device model management, image generation, and chat.
  XcodeGen owns the project (`apps/ios/project.yml`), with the
  `MereRunStudio` app, the `MereRunWidgets` extension, and the
  `MereRunStudioTests` and `MereRunStudioUITests` bundles. CI regenerates the
  project and builds it for the simulator in the Release configuration.

## Phasing

- **Phase A: run inbox and prompt-first submission (landed).** Pair, watch,
  fetch, cancel and retry, fleet, and Create. The graph documents, node
  registry, validator, and bundle materializer live in `MereRunRelayKit`, so
  the phone materializes byte-identical bundles and submits through the same
  create, upload, and commit path as the CLI. The Create surface generates its
  forms from the shared node catalog (image, video, music, SFX, and speech
  synthesis), reports per-kind fleet availability from the worker probe, and
  pins models from the fleet's installed list. Provider-qualified nodes stay
  excluded on iOS, and modes whose required inputs are assets wait for the
  photo and file picker step.
- **Phase B: chat (landed).** The app routes chat to the selected transport:
  on-device small models when present, a directly paired machine, or relay
  jobs otherwise. The **Connect to a machine** pairing uses
  `mere.run relay serve`, which serves the full job vocabulary over a LAN or
  tailnet with no cloud hop. It supersedes the earlier idea of running
  `api serve` over a LAN.

  Token streaming through the relay is an additive event-type revision, not an
  architecture change. The node-to-relay leg is a persistent WebSocket, the
  relay serves `/events` SSE-framed, and the implementation treats the event
  `type` as an open string. Because `graph-event-v1` is a closed schema, a
  `node_output_delta` event needs a coordinated contract revision with the
  relay and node repositories' shared fixtures.
- **Phase C: on-device models (experimental).** Device RAM and available
  storage gate the managed Bonsai and Liquid models. Model payloads use one
  fixed, system-owned background session, so the active file can continue
  while the app is suspended. The app durably records the requested install,
  rejoins matching system tasks after relaunch, moves completed payloads out of
  the URLSession temporary directory, and then resumes the pinned snapshot and
  final validation pipeline. A user-initiated download defaults to unmetered,
  unconstrained Wi-Fi. You can explicitly allow cellular, expensive, and
  constrained networks before you start a download. The app records that
  choice with the pending install, so a relaunch can't silently change the
  transfer policy.

## On-device lifecycle

- Chat warmup is immediate. The app releases resident chat or image state after
  two idle minutes, when the selected chat model or execution lane changes,
  when the app enters the background, and on an iOS memory warning.
- A lifecycle release doesn't interrupt active inference. The app defers it
  until the active image or chat request finishes, and the engine rejects
  another request while the release is in progress.
- When you remove an installed model, the app unloads all runtime state before
  it deletes the install. It reclaims only cache payloads that no other model
  references.
- iOS might relaunch the app to deliver background-session events. The app
  reconnects that session through its application delegate and resumes every
  durable pending install. iOS controls what happens after you force-quit the
  app, and the app doesn't treat that as a supported continuation path.
- Each job has one owner for Live Activity polling. Transient request failures
  use bounded exponential backoff. Persistent failures end the activity instead
  of leaving a poller running forever. Unpairing cancels all outstanding
  pollers.
- Refreshing a run's artifacts stages and validates the complete replacement
  result beside the existing result, and then atomically replaces it. A failed
  refresh leaves the last verified local copy intact.
- Privacy copy follows the selected execution lane. On-device work stays on the
  phone, the direct-machine lane names that machine connection, and relay work
  describes the authenticated relay and fleet path instead of claiming that
  every remote request is local-only.

## Follow-up work

- Keychain-backed credential storage and Authorization Code with PKCE sign-in
  are done. Sign-in uses `ASWebAuthenticationSession` with the broker's
  advertised `authorization_endpoint` and a universal-link callback on
  `mere.world`. Tokens live in the Keychain, with a one-time migration from
  file storage. The device grant remains the browserless fallback.
- Artifact downloads stream to disk with digest verification. Incremental
  in-flight hashing is planned.
- Physical-device interruption coverage still needs a release-signed run that
  backgrounds the app during a real multi-gigabyte model pull, verifies
  relaunch reattachment, and observes runtime memory returning after eviction.
- Debug builds use the associated-domain developer alternate mode; Release
  builds use the production AASA CDN. The app and widget versions must stay
  identical, and CI checks both.
- Live Activities ship with app-driven updates. Completion pushes and
  push-updated activities need APNs support in the relay repository.

## Release footprint

A clean arm64 Release simulator build on 2026-08-19 measured 75.4 MiB
uncompressed, including a 68.3 MiB main executable and a 4.3 MiB MLX Metal
library. The llama.cpp binary dependency is macOS-only, so it isn't in the
iOS build. This isn't an App Store download-size measurement, but it confirms
that the broad `MereRunCore` closure is the main compile-time and binary-size
cost.

CI rejects a simulator executable that isn't arm64-only and caps that
executable at 80 MiB. Extracting the image, chat, and model storage surface
into a narrower mobile runtime remains a separate architectural optimization.
A facade that still depends on `MereRunCore` wouldn't reduce the dependency
closure.

## Boundaries

The wire shapes in `MereRunRelayKit` are the published contract. If the iOS app
needs different fields, event types, or similar shapes, revise the
cross-repository contract before you add the app feature. The app doesn't use a
private schema.
