import Foundation

/// Which depth parameterization a Marigold V2 checkpoint was trained to emit.
///
/// Log and linear depth both increase with distance; disparity decreases. The
/// distinction is not cosmetic: it decides which end of the decoded range is
/// near the camera, so previews and any downstream ordering depend on it.
public enum MarigoldV2DepthParameterization: String, Codable, CaseIterable, Sendable {
    case log
    case linear
    case disparity

    /// True when larger decoded values mean closer geometry.
    public var increasesTowardCamera: Bool {
        self == .disparity
    }
}

/// A published Marigold V2 depth checkpoint.
///
/// Every depth checkpoint shares one precomputed prompt, so the variant only
/// selects the adapter directory and how the decoded values are read.
public enum MarigoldV2DepthCheckpoint: String, Codable, CaseIterable, Sendable {
    case logStage2 = "log-stage2"
    case logStage1 = "log-stage1"
    case logLayered = "log-layered"
    case uniformBase = "uniform-base"
    case uniformLayered = "uniform-layered"
    case disparityBase = "disparity-base"
    case disparityLayered = "disparity-layered"

    /// Path of the checkpoint directory inside the Marigold repository.
    public var relativeDirectory: String {
        switch self {
        case .logStage2: "depth/Log-stage2"
        case .logStage1: "depth/Log-stage1"
        case .logLayered: "depth/Log-layered"
        case .uniformBase: "depth/Uniform-base"
        case .uniformLayered: "depth/Uniform-layered"
        case .disparityBase: "depth/Disparity-base"
        case .disparityLayered: "depth/Disparity-layered"
        }
    }

    public var parameterization: MarigoldV2DepthParameterization {
        switch self {
        case .logStage2, .logStage1, .logLayered: .log
        case .uniformBase, .uniformLayered: .linear
        case .disparityBase, .disparityLayered: .disparity
        }
    }

    /// See-through variants predict the geometry behind transparent surfaces
    /// rather than the first surface the ray meets.
    public var isSeeThrough: Bool {
        switch self {
        case .logLayered, .uniformLayered, .disparityLayered: true
        case .logStage2, .logStage1, .uniformBase, .disparityBase: false
        }
    }

    /// Checkpoints whose recipe fine-tuned the VAE decoder ship `VAE.*` tensors.
    /// The remaining ones decode with the frozen base decoder.
    public var shipsFineTunedVAEDecoder: Bool {
        switch self {
        case .logStage2, .logLayered, .disparityBase, .disparityLayered: true
        case .logStage1, .uniformBase, .uniformLayered: false
        }
    }
}

/// Marigold V2 repurposes a frozen Qwen-Image-Edit-2509 diffusion transformer
/// into a single-step dense predictor.
///
/// A published checkpoint is one `trainables.safetensors` carrying rank-128 LoRA
/// adapters for the transformer under `Diffuser.*` and, where the recipe trained
/// one, a replacement VAE decoder under `VAE.*`. Prompt conditioning ships
/// precomputed, so the base text encoder is never loaded and never downloaded.
public enum MarigoldV2Repository {
    public static let modelId = "vision-depth-marigold-v2"
    public static let upstreamRepoId = "huawei-bayerlab/marigold-v2-0"
    public static let upstreamRevision = "6fd6d1ca246c9d2d99a4d8ac375a4eccc87178ad"
    public static let license = "Apache-2.0"

    /// Marigold V2 is trained against Qwen-Image-Edit-2509. The 2511 checkpoint
    /// already in the catalog is a different base and is not interchangeable.
    public static let baseRepoId = "Qwen/Qwen-Image-Edit-2509"
    public static let baseRevision = "d3968ef930e841f4c73640fb8afa3b306a78167e"

    /// The base subset this runtime executes. `text_encoder/`, `tokenizer/`, and
    /// `processor/` are deliberately absent: the precomputed prompt embeddings
    /// replace them and keep 16.6 GB of encoder weights out of the install.
    public static let basePatterns = [
        "model_index.json",
        "scheduler/*",
        "transformer/*",
        "vae/*",
    ]

    /// Directory inside the managed install where the Marigold payload is mounted.
    public static let adapterMountPath = "marigold"
    public static let promptEmbeddingsDirectory = "qwen_text_embeddings"

    /// The managed install ships the paper checkpoint. Other variants resolve
    /// from a local Marigold repository root passed on the command line.
    public static let installedCheckpoint = MarigoldV2DepthCheckpoint.logStage2

    /// Depth conditioning is one precomputed prompt shared by every depth checkpoint.
    public static let depthPromptPrefix = "qwen_edit_2509_qwen_depth_realimg512"

    public static let adapterPatterns = [
        "\(installedCheckpoint.relativeDirectory)/trainables.safetensors",
        "\(promptEmbeddingsDirectory)/\(depthPromptPrefix)_prompt_embeds.pt",
        "\(promptEmbeddingsDirectory)/\(depthPromptPrefix)_prompt_mask.pt",
        "LICENSE*",
        "manifest.json",
    ]

    public static let baseHubFallback = HubFallbackConfig(
        repoId: baseRepoId,
        revision: baseRevision,
        patterns: basePatterns
    )

    public static let adapterHubFallback = HubFallbackConfig(
        repoId: upstreamRepoId,
        revision: upstreamRevision,
        patterns: adapterPatterns
    )

    public static let mountedHubFallbacks = [
        MountedHubFallbackConfig(
            destinationPath: adapterMountPath,
            hubFallback: adapterHubFallback
        ),
    ]

    /// Base subset plus the shipped depth checkpoint and its prompt embeddings.
    public static let estimatedDownloadBytes: Int64 = 42_988_353_807

    public static let trainablesPin = ModelArtifactPin(
        filename: "\(adapterMountPath)/\(installedCheckpoint.relativeDirectory)/trainables.safetensors",
        byteCount: 1_853_909_694,
        sha256: "3edec69490ba8e8fdc8d63e130f93cb93f514360d415a22701250b2535056892"
    )

    public static let promptEmbedsPin = ModelArtifactPin(
        filename: "\(adapterMountPath)/\(promptEmbeddingsDirectory)/\(depthPromptPrefix)_prompt_embeds.pt",
        byteCount: 19_384_278,
        sha256: "ba752b582571d990c75ad6aac2a2eeb0cf04c2fe01b233b1675028eabf342a6d"
    )

    public static let promptMaskPin = ModelArtifactPin(
        filename: "\(adapterMountPath)/\(promptEmbeddingsDirectory)/\(depthPromptPrefix)_prompt_mask.pt",
        byteCount: 23_624,
        sha256: "80973a6521f21a5a6faf6402cae185dafef840d91e654efbeb1097914e634798"
    )

    public static let artifactPins = [trainablesPin, promptEmbedsPin, promptMaskPin]

    /// Verifies the pinned Marigold payload of a managed install.
    ///
    /// Only the managed layout is pinned. A Marigold repository the caller
    /// points at directly carries no pin, so verification is skipped for it and
    /// the loaders fall back to shape and count checks.
    public static func verifyInstalledArtifacts(
        rootURL: URL,
        fileManager: FileManager = .default
    ) throws {
        let mounted = rootURL.appending(path: adapterMountPath, directoryHint: .isDirectory)
        guard fileManager.fileExists(atPath: mounted.path) else {
            return
        }
        for pin in artifactPins {
            _ = try pin.verify(in: rootURL, fileManager: fileManager)
        }
    }
}

/// File layout of an installed Marigold V2 depth model.
///
/// The root holds the frozen Qwen-Image-Edit-2509 subset; `marigold/` holds the
/// adapter payload mounted from the Marigold repository.
public struct MarigoldV2Resources: Sendable, Hashable {
    public var rootURL: URL
    public var checkpoint: MarigoldV2DepthCheckpoint

    public init(rootURL: URL, checkpoint: MarigoldV2DepthCheckpoint = MarigoldV2Repository.installedCheckpoint) {
        self.rootURL = rootURL
        self.checkpoint = checkpoint
    }

    // MARK: - Frozen base

    public var modelIndexURL: URL {
        rootURL.appending(path: "model_index.json")
    }

    public var transformerConfigURL: URL {
        rootURL.appending(path: "transformer/config.json")
    }

    public var transformerWeightsIndexURL: URL {
        rootURL.appending(path: "transformer/diffusion_pytorch_model.safetensors.index.json")
    }

    public var transformerWeightsURL: URL {
        rootURL.appending(path: "transformer/diffusion_pytorch_model.safetensors")
    }

    public var quantizedTransformerWeightsURL: URL {
        rootURL.appending(path: "transformer/model.safetensors")
    }

    public var vaeConfigURL: URL {
        rootURL.appending(path: "vae/config.json")
    }

    public var vaeWeightsURL: URL {
        rootURL.appending(path: "vae/diffusion_pytorch_model.safetensors")
    }

    // MARK: - Marigold payload

    /// Root of the mounted Marigold repository. A local Marigold checkout passed
    /// directly on the command line has the same shape without the mount prefix,
    /// so both layouts resolve through `adapterRootURL`.
    public var adapterRootURL: URL {
        let mounted = rootURL.appending(
            path: MarigoldV2Repository.adapterMountPath,
            directoryHint: .isDirectory
        )
        return FileManager.default.fileExists(atPath: mounted.path) ? mounted : rootURL
    }

    public var trainablesURL: URL {
        adapterRootURL.appending(path: "\(checkpoint.relativeDirectory)/trainables.safetensors")
    }

    public var promptEmbedsURL: URL {
        adapterRootURL.appending(
            path: "\(MarigoldV2Repository.promptEmbeddingsDirectory)/"
                + "\(MarigoldV2Repository.depthPromptPrefix)_prompt_embeds.pt"
        )
    }

    public var promptMaskURL: URL {
        adapterRootURL.appending(
            path: "\(MarigoldV2Repository.promptEmbeddingsDirectory)/"
                + "\(MarigoldV2Repository.depthPromptPrefix)_prompt_mask.pt"
        )
    }

    // MARK: - Validation

    /// Files that must exist before the runtime will attempt to load.
    ///
    /// The text encoder and tokenizer are intentionally not checked: Marigold
    /// conditions on precomputed embeddings and never instantiates them.
    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []
        for url in [modelIndexURL, transformerConfigURL, vaeConfigURL, vaeWeightsURL] {
            if !fileManager.fileExists(atPath: url.path) {
                missing.append(url)
            }
        }

        let transformerWeightsPresent = fileManager.fileExists(atPath: transformerWeightsIndexURL.path)
            || fileManager.fileExists(atPath: transformerWeightsURL.path)
            || fileManager.fileExists(atPath: quantizedTransformerWeightsURL.path)
        if !transformerWeightsPresent {
            missing.append(transformerWeightsIndexURL)
        }

        for url in [trainablesURL, promptEmbedsURL, promptMaskURL] {
            if !fileManager.fileExists(atPath: url.path) {
                missing.append(url)
            }
        }
        return missing
    }
}
