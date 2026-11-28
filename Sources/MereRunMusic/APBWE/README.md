# AP-BWE audio enhancement

This module is a native Swift/MLX port of the MIT-licensed AP-BWE 16 kHz to
48 kHz speech bandwidth-extension profile.

- `APBWEConfiguration.swift` owns the exact typed profile.
- `APBWEResources.swift` pins source, transport snapshot, checkpoint, config,
  code license, and weights license.
- `APBWEModel.swift` owns the magnitude/phase ConvNeXt graph and restricted
  PyTorch state-dict loader.
- `APBWEEnhancer.swift` owns deterministic 3x interpolation and overlap-add
  chunk execution.

The managed checkpoint is downloaded separately. Do not relax the artifact
pins or graph-inventory checks to admit a repackaged checkpoint.
