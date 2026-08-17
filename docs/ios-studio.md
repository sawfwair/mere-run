# iOS Studio

The iOS app (`ios/`) is the third thin client in the family that hosted Graph
Studio established: it signs in to a relay, submits and watches work, and
fetches verified artifacts, while paired nodes own model execution. This
document records the architecture decisions and phasing; `ios/README.md`
covers building.

## Two lanes

1. **Relay lane (primary).** Full capability comes from the paired fleet: the
   phone speaks the same authenticated HTTPS + JSON job API as the CLI's
   relay executor, gets its node catalog from the fleet through relay (the
   hosted-Studio pattern — never from local discovery, which needs process
   spawning that iOS forbids), and reads the same `graph-event-v1` /
   `graph-run-v1` documents published in `docs/public/schemas/`.
2. **On-device lane (later).** Small models genuinely fit Pro-class devices:
   the Bonsai image binary (4B, 1-bit, ~3.4 GB) runs through the same FLUX.2
   Klein pipeline that already has a memory-constrained iOS generator in
   `MereRunCore` (`Flux2KleinGeneratoriOS`, ~2 GB peak via sequential
   loading), and `text-chat-bonsai-27b-1bit` (~5.1 GB, Apache-2.0,
   vision-capable) is plausible on 12 GB+ devices with the increased-memory
   entitlement. This lane requires an iOS build gate for a trimmed
   `MereRunCore` (ONNX paths are already macOS-conditional; the `IOKit`
   linker setting needs a platform condition) and a storage/download UX, and
   is explicitly out of scope for the first shippable phase.

## What exists today

- `Sources/MereRunRelayKit` — the portable client half of relay execution,
  extracted from `MereRunCLI`: executor profiles and references, OAuth
  device-grant authentication, the graph-job/fleet HTTP client with verified
  artifact fetch, the workflow wire types, and SSE event-text normalization.
  Foundation + swift-crypto only; builds and tests on Linux; the CLI consumes
  it with unchanged behavior.
- `ios/` — the SwiftUI scaffold: pairing via device grant, fleet view, run
  inbox, run detail with polled events/progress, cancel/retry, and artifact
  fetch with share. Generated with XcodeGen; first Xcode build still pending
  (authored in a Linux environment).

## Phasing

- **Phase A — run inbox (current).** Pair, watch, fetch, cancel/retry,
  fleet. Add prompt-first submission next: JSON-value inputs plus photo
  upload need the bundle-materialization layer (graph documents, fingerprint
  hashing, asset manifests) extracted from `MereRunCLI` the same way the
  relay client was; provider-qualified nodes stay excluded on iOS.
- **Phase B — chat.** Route chat to the best available transport: on-device
  small models when present, a Mac's `api serve` over the LAN when reachable
  (real SSE token streaming today), relay jobs otherwise. Token streaming
  *through relay* is an additive event-type revision, not an architecture
  change: the node→relay leg is a persistent WebSocket, relay serves
  `/events` SSE-framed, and the event `type` is an open string in the
  implementation — but `graph-event-v1` is a closed schema, so a
  `node_output_delta` event needs a coordinated contract revision with the
  relay and node repositories' shared fixtures.
- **Phase C — on-device tiny models.** The Bonsai lane above, gated by
  device RAM, Wi-Fi-only multi-GB pulls, and a storage manager.

## iOS-specific follow-ups

- Keychain-backed credential storage and Authorization Code + PKCE sign-in
  (hosted Studio's flow; a phone has a browser, so the device grant is a
  working but non-idiomatic first step).
- Streaming artifact downloads (`URLSession.downloadTask` with incremental
  SHA-256) before video/3D outputs; the current fetch buffers whole
  artifacts in memory.
- Completion pushes and Live Activities for long renders need APNs support
  in the relay repository; until then the app polls in the foreground and
  can adopt background refresh.

## Boundaries

The wire shapes in `MereRunRelayKit` are the published contract. Anything the
iOS app needs that would change those shapes — new event types, new job
fields — is a cross-repository contract revision first, an app feature
second. The app never gains a private schema.
