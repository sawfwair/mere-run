# Cosmos3 native runtime

This directory contains the native Swift/MLX ports of NVIDIA Cosmos3-Edge and
the distilled Cosmos3-Super text-to-image model. Cosmos3-Edge uses the official,
immutable Diffusers snapshot directly. Cosmos3-Super uses a publishable MLX Q4
artifact converted from an immutable NVIDIA revision. Neither runtime launches
Python or uses remote inference.

## Reading order

1. `Cosmos3Resources.swift` defines the pinned snapshot layout and published
   transformer, VAE, and reasoner configurations.
2. `Cosmos3Action.swift` defines the 15 published action domains, camera
   rotation-6D encoding, and forward/policy/inverse action contracts.
3. `Cosmos3Sequence.swift` implements text, vision, and action packing plus
   multimodal rotary position IDs.
4. `Cosmos3Transformer.swift` implements the shared omnimodal
   generation/understanding transformer, including the Edge ReLU-squared path
   and the Super gated-SiLU path.
5. `Cosmos3ModelLoader.swift` maps the official Diffusers checkpoint into the
   native transformer and shared Wan VAE.
6. `Cosmos3Tokenizer.swift` reproduces the published generation prompts and
   multimodal metadata.
7. `Cosmos3Scheduler.swift` implements NVIDIA's shifted-flow UniPC schedule and
   the stochastic four-step Euler schedule published with Cosmos3-Super.
8. `Cosmos3EdgeGenerator.swift` owns text/image/video generation, image
   editing, action-conditioned forward dynamics, policy generation, inverse
   dynamics, decode, and output.
9. `Cosmos3ReasonerVision.swift` and `Cosmos3Reasoner.swift` implement the
   packed SigLIP2 vision path, projector, multimodal rotary layout, and
   understanding decode.
10. `Cosmos3CameraTrajectory.swift` loads NVIDIA's pinned normalized 60x9
    `camera_pose` reference path and compiles semantic world controls.
11. `Cosmos3WorldSession.swift` keeps the model resident, re-encodes each
    terminal public frame, and continues the normalized camera trajectory
    across action-conditioned world transitions while advancing NVIDIA's
    `base_seed + chunk_index` autoregressive sampling sequence. Exact inverse
    controls pop and traverse cached parent edges backward instead of
    regenerating them.

## Public entrypoints

- CLI generation and understanding:
  `Sources/MereRunCLI/Commands/VideoCosmos3Command.swift`
- Persistent world HTTP runtime:
  `Sources/MereRunCLI/Commands/WorldCommand.swift`
- User guide: `Sources/MereRunCLI/Guides/video-cosmos3.md`
- Runtime docs: `docs/runtime/video.md` and `docs/runtime/world.md`

## Parity evidence

`Tests/MereRunCoreTests/Cosmos3EdgeTests.swift` covers both transformer families,
published configuration and checkpoint inventories, the distilled schedule,
action layouts, prompt templates, sequence geometry, mode defaults, warm-world
continuity, and deterministic Edge fixtures exported from the pinned NVIDIA
source by `scripts/export-cosmos3-edge-parity.py`.
