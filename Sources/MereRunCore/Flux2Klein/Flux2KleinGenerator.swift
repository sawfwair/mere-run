import Foundation
import MLX
import MLXRandom
import MLXNN

// MARK: - Flux2 Klein Generator

/// Generator for FLUX.2 Klein image generation
public actor Flux2KleinGenerator: ImageGenerator {

    // MARK: - State

    var transformer: Flux2Transformer2DModel?
    var textEncoder: QwenTextEncoder?
    var tokenizer: QwenTokenizer?
    var vae: AutoencoderKL?
    var bnRunningMean: MLXArray?
    var bnRunningVar: MLXArray?
    var loadedModelPath: String?
    var loadedManifest: MereRunModelManifest?
    var loadedQuantization: ModelWeightsLoader.QuantizationParams?
    var currentLoRAs: [LoRA] = []
    var currentTextLoRA: LoRA?
    var transformerLoRALayers: [String: TrainableLoRALayer]?
    var transformerLoRARankSignature: String?
    var compiledTransformer: (@Sendable ([MLXArray]) -> [MLXArray])?
    var compiledTransformerNeedsWarmup: Bool = true

    public init() {}

    /// Preserve the checkpoint filename when its payload is a content-addressed symlink.
    /// MLX selects the loader from the visible `.safetensors` extension.
    static func checkpointFileURL(in directoryURL: URL, filename: String) -> URL {
        directoryURL.appendingPathComponent(filename)
    }

    // MARK: - Memory Management

    /// Unload all models from memory to free GPU/RAM
    public func unload() {
        transformer = nil
        textEncoder = nil
        tokenizer = nil
        vae = nil
        bnRunningMean = nil
        bnRunningVar = nil
        loadedModelPath = nil
        loadedManifest = nil
        loadedQuantization = nil
        currentLoRAs = []
        transformerLoRALayers = nil
        transformerLoRARankSignature = nil
        currentTextLoRA = nil
        compiledTransformer = nil

        Memory.clearCache()
    }

    // MARK: - Unified ImageGenerator API

    public func generate(
        _ request: GenerationRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult {
        let modelSpec = request.model ?? ModelResolver.ModelID.kleinNano.rawValue
        let modelPath = try resolveModelPath(from: modelSpec)

        let result = try await generate(request, modelPath: modelPath, progressHandler: progressHandler)
        progressHandler?(GenerationProgress(stage: .saving, stepIndex: 1, totalSteps: 1))
        return result
    }

    private func resolveModelPath(from spec: String) throws -> String {
        let fm = FileManager.default
        var isDir: ObjCBool = false

        let localURL = URL(fileURLWithPath: spec).standardizedFileURL
        if fm.fileExists(atPath: localURL.path, isDirectory: &isDir), isDir.boolValue {
            return localURL.path
        }

        if let id = ModelResolver.ModelID(rawValue: spec) {
            return try ModelResolver().resolve(id).rootURL.path
        }

        throw Flux2Error.modelNotFound(spec)
    }

}

// MARK: - Errors

public enum Flux2Error: LocalizedError {
    case modelsNotLoaded
    case modelNotFound(String)
    case invalidOutputDirectory(URL)
    case insufficientHiddenStates
    case invalidLatentShape
    case imageSaveFailed
    case missingBatchNormStats
    case referenceImageNotFound(URL)
    case referenceImageDecodeFailed(URL)
    case invalidManifest(String)
    case invalidSigmaSchedule(String)

    public var errorDescription: String? {
        switch self {
        case .modelsNotLoaded:
            return "Models not loaded"
        case .modelNotFound(let spec):
            return "Model not found: \(spec)"
        case .invalidOutputDirectory(let url):
            return "Output directory does not exist: \(url.deletingLastPathComponent().path)"
        case .insufficientHiddenStates:
            return "Text encoder did not return enough hidden states"
        case .invalidLatentShape:
            return "Invalid latent tensor shape"
        case .imageSaveFailed:
            return "Failed to save image"
        case .missingBatchNormStats:
            return "VAE weights missing BatchNorm running statistics"
        case .referenceImageNotFound(let url):
            return "Reference image not found: \(url.path)"
        case .referenceImageDecodeFailed(let url):
            return "Failed to decode reference image: \(url.path)"
        case .invalidManifest(let message):
            return "Invalid model manifest: \(message)"
        case .invalidSigmaSchedule(let message):
            return "Invalid FLUX.2 sigma schedule: \(message)"
        }
    }
}
