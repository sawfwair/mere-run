# Third-Party Notices

This repo includes and depends on third-party software.

Bundled artifact:

- `vendor/llama.xcframework` is built from `llama.cpp`
- `vendor/mlx-swift_Cmlx.bundle` is the bundled MLX shader resource used by the local package build

Managed model archives may also include third-party model weights. For the native
vision segmentation runtime, `vision-segment-sam31` is sourced from the upstream
Meta SAM 3.1 release (`facebook/sam3.1`). Its license terms, usage conditions,
and notices remain with the upstream model release.

Direct Swift package dependencies:

- `mlx-swift`
- `swift-transformers`
- `swift-argument-parser`
- `hummingbird`

Their licenses and notices remain with their upstream projects and resolved package checkouts.
