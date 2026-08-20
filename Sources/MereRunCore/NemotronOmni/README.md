# Nemotron Omni runtime

This module is the native Swift/MLX execution engine for NVIDIA's pinned
Nemotron 3 Nano Omni 30B-A3B Reasoning BF16 checkpoint. It accepts text,
images, local audio, and local video and generates text without Python,
Transformers, vLLM, or an intermediate model conversion.

The runtime has four layers:

- `NemotronOmniGenerator` owns model lifetime, prompt/media ordering,
  multimodal prefill, and autoregressive generation.
- `NemotronOmniVisionModel` and the image/video processors implement the
  C-RADIO v4-H vision tower, dynamic image tiling, and two-frame video
  tubelets.
- `NemotronOmniSoundModel` and the audio processor implement the Parakeet
  log-Mel frontend, convolutional subsampling, and conformer encoder.
- `NemotronOmniLanguageModel` implements the BF16 Nemotron-H hybrid
  attention/Mamba/MoE backbone. `NemotronOmniExpertPack` creates one
  source-bound, stacked expert cache beside the source snapshot so the 46
  routed expert tensors do not require 17-shard gathers on every launch.
  Multitoken prefill evaluates at layer boundaries to keep every Metal command
  buffer below the macOS watchdog; one-token decode remains fused.

Managed installation remains explicit because the upstream checkpoint uses
the NVIDIA Open Model Agreement. The small internal MereRun wrapper may point
at an immutable snapshot on external storage; the 66 GB source checkpoint and
58 GB derived expert cache must not be duplicated merely to make the catalog
entry visible.

Media inputs are decoded locally. Image and video URLs passed to this runtime
must be local paths or `file:` URLs. Audio uses the shared local audio loader.
PDF documents are handled by rendering pages to images before inference; PDF
is not a separate model input modality.

When changing this module, run the focused `NemotronOmniTests`, the repository
gate, and at least one real installed-checkpoint generation for every media
tower affected by the change.
