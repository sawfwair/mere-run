import Foundation

public enum Krea2RawResources {
    public static let modelId = "image-krea2-raw"
    public static let upstreamRepoId = "krea/Krea-2-Raw"
    public static let upstreamRevision = "4ad9f4b627a647fad78b3dfeebb09f2654aeb494"
    public static let estimatedDownloadBytes: Int64 = 36 * 1_073_741_824

    /// Raw ships both a root-level `raw.safetensors` file for Krea's official
    /// codebase and the Diffusers component layout used by mere.run. Pull the
    /// component layout only so managed installs do not download the transformer
    /// twice.
    public static let snapshotPatterns = [
        "LICENSE.pdf",
        "README.md",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "tokenizer/chat_template.jinja",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "transformer/diffusion_pytorch_model-*.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]
}
