# Nemotron Image runtime

`NemotronImageGenerator` tokenizes the prompt, samples a grid with the shared
Nemotron-H language model, and passes the grid to the L24SQ decoder in
`MereRunImageModels`. The generator and decoder load in separate stages.

`NemotronImageLoader` reads NVIDIA's indexed FP32 shards or a local packed
checkpoint. Packed checkpoints store one safetensors file per language block,
with separate files for the embedding table, final norm, and output head.
The manifest records the bit width, group size, and packing scheme.

`NemotronImageQuantization` implements `model optimize` for this family.
It packs only the routed expert matrices. Prefill and single-token decode
both use packed expert multiplication. The VisualInference Q8 import path
maps the existing affine weights, scales, and biases to the Swift module keys.

For conversion and generation commands, see the [image runtime guide](../../../docs/runtime/image.md).
