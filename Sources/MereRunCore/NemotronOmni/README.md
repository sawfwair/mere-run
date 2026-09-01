# Nemotron Omni runtime

This module is the native Swift/MLX execution engine for the pinned standalone
Nemotron 3 Nano Omni 30B-A3B Reasoning BF16 MLX checkpoint. It accepts text,
images, local audio, and local video and generates text without Python,
Transformers, vLLM, or a first-run model conversion.

The runtime has four layers:

- `NemotronOmniGenerator` owns model lifetime, prompt/media ordering,
  multimodal prefill, and autoregressive generation.
- `NemotronOmniVisionModel` and the image/video processors implement the
  C-RADIO v4-H vision tower, dynamic image tiling, and two-frame video
  tubelets.
- `NemotronOmniSoundModel` and the audio processor implement the Parakeet
  log-Mel frontend, convolutional subsampling, and conformer encoder.
- `NemotronOmniLanguageModel` implements the BF16 Nemotron-H hybrid
  attention/Mamba/MoE backbone. `NemotronOmniExpertPack` loads the published
  stacked expert shard directly so the 46 routed expert tensors do not require
  17-shard gathers on every launch. The non-expert shards retain their upstream
  keys, shapes, dtypes, and payload bytes.
  Multitoken prefill evaluates at layer boundaries to keep every Metal command
  buffer below the macOS watchdog; one-token decode remains fused.

Managed installation remains explicit because the checkpoint uses the NVIDIA
Open Model Agreement. The managed pull downloads one approximately 66 GB
artifact containing every model parameter exactly once. A legacy source
snapshot with a locally generated expert cache remains readable, but is no
longer the managed installation path.

Media inputs are decoded locally. Image and video URLs passed to this runtime
must be local paths or `file:` URLs. Audio uses the shared local audio loader.
PDF documents are handled by rendering pages to images before inference; PDF
is not a separate model input modality.

When changing this module, run the focused `NemotronOmniTests`, the repository
gate, and at least one real installed-checkpoint generation for every media
tower affected by the change.
