# UniverSR native runtime

This directory contains the native Swift/MLX port of the official UniverSR
general-audio checkpoint. It does not launch Python, PyTorch, or CUDA.

- `UniverSRConfiguration.swift` defines the typed, frozen graph and inference
  profile corresponding to the pinned upstream YAML.
- `UniverSRResources.swift` independently pins the MIT code revision and the
  CC BY 4.0 checkpoint snapshot, including exact artifact hashes.
- `UniverSRModel.swift` owns the conditional ConvNeXt V2 U-Net and restricted
  PyTorch state-dict loader.
- `UniverSREnhancer.swift` owns STFT compression, deterministic flow-matching
  ODE integration, chunking, and waveform reconstruction.

The bundled JSON is runtime configuration, not a copy of the upstream YAML.
The original `config.yaml`, checkpoint, and model-card license declaration are
downloaded and verified byte-for-byte by the managed model resolver.
