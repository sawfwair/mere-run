import Foundation
import MLX
import MLXRandom
import MLXNN

// MARK: - Flux2 Klein Generator (iOS Memory-Optimized)

/// Memory-optimized generator for FLUX.2 Klein on iOS
/// Loads models sequentially to reduce peak memory from ~4.2GB to ~2GB
public actor Flux2KleinGeneratoriOS: ImageGenerator {

    // Tiny state (kept loaded - ~2MB total)
    var tokenizer: QwenTokenizer?
    var bnRunningMean: MLXArray?
    var bnRunningVar: MLXArray?
    var loadedModelPath: String?
    var loadedManifest: MereRunModelManifest?

    public init() {
        // Use moderate cache for iOS - too small can cause issues
        // 256MB should allow buffer reuse while staying under memory pressure
        Memory.cacheLimit = 256 * 1024 * 1024
    }

    /// Clear GPU cached memory
    func clearGPUMemory(synchronize: Bool = true) {
        // Clearing while kernels are in-flight can lead to corrupted outputs (e.g. blurry images).
        if synchronize {
            Stream.gpu.synchronize()
        }
        Memory.clearCache()
    }

    // MARK: - Generation

    public func generate(
        _ request: GenerationRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult {
        let seed = request.seed ?? UInt64.random(in: 0..<UInt64.max)
        let modelSpec = request.model ?? ModelResolver.ModelID.kleinNano.rawValue
        let modelPath = try resolveModelPath(from: modelSpec)

        var resolved = request
        resolved.seed = seed

        let outputURL = try await generate(resolved, modelPath: modelPath, progressHandler: progressHandler)
        progressHandler?(GenerationProgress(stage: .saving, stepIndex: 1, totalSteps: 1))
        return GenerationResult(outputURL: outputURL, seed: seed)
    }

    func generate(
        _ request: GenerationRequest,
        modelPath: String,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> URL {
        // Load persistent state (tokenizer + BN stats) if needed
        if loadedModelPath != modelPath {
            try await loadPersistentState(from: modelPath)
        }

        guard let tokenizer = tokenizer,
              let bnMean = bnRunningMean,
              let bnVar = bnRunningVar else {
            throw Flux2Error.modelsNotLoaded
        }

        guard let manifest = loadedManifest else {
            throw Flux2Error.invalidManifest("Missing \(MereRunModelManifest.filename)")
        }

        let modelRootURL = URL(fileURLWithPath: modelPath).standardizedFileURL
        let componentResolver = ModelComponentResolver(modelRootURL: modelRootURL, manifest: manifest)
        let transformerComponent = try componentResolver.resolveDirectory(for: .transformer, fallbackLocalPath: "transformer")
        let textEncoderComponent = try componentResolver.resolveDirectory(for: .textEncoder, fallbackLocalPath: "text_encoder")
        let vaeComponent = try componentResolver.resolveDirectory(for: .vae, fallbackLocalPath: "vae")

        guard let variant = manifest.variant else {
            throw Flux2Error.invalidManifest("Missing manifest.variant")
        }
        let isDistilled = variant == .distilled

        let transformerQuantization = try ModelWeightsLoader.QuantizationParams.fromManifest(transformerComponent.sourceManifest)
        let textEncoderQuantization = try ModelWeightsLoader.QuantizationParams.fromManifest(textEncoderComponent.sourceManifest)

        // 1. Prepare latent dimensions
        let seed = request.seed ?? UInt64.random(in: 0..<UInt64.max)
        let vaeScaleFactor = 8  // Standard for FLUX VAE
        let latentHeight = request.height / vaeScaleFactor
        let latentWidth = request.width / vaeScaleFactor
        let patchedHeight = latentHeight / 2
        let patchedWidth = latentWidth / 2
        let seqLen = patchedHeight * patchedWidth

        // 2. Encode reference images if any (loads VAE temporarily)
        var referenceLatents: [MLXArray] = []
        let referenceImages = Array(request.referenceImages.prefix(4))
        let numRefs = referenceImages.count
        if numRefs > 0 {
            referenceLatents = try await encodeReferenceImages(
                urls: referenceImages,
                width: request.width,
                height: request.height,
                patchedHeight: patchedHeight,
                patchedWidth: patchedWidth,
                bnMean: bnMean,
                bnVar: bnVar,
                referenceStrength: Float(request.referenceStrength),
                seed: seed,
                vaeDirURL: vaeComponent.directoryURL,
                progressHandler: progressHandler
            )

            // VAE fully deallocated now - clear cache before next stage loads
            clearGPUMemory()
            await Task.yield()
        }

        // 3. Encode prompts and save to disk (ensures text encoder is fully freed before transformer)
        let embeddingsURL = FileManager.default.temporaryDirectory.appendingPathComponent("prompt_\(seed).npy")
        let negEmbeddingsURL = FileManager.default.temporaryDirectory.appendingPathComponent("neg_\(seed).npy")
        let hasNegative = try await encodePromptsAndSave(
            prompt: request.prompt,
            negativePrompt: request.negativePrompt,
            guidanceScale: Float(request.guidanceScale),
            tokenizer: tokenizer,
            textEncoderDirURL: textEncoderComponent.directoryURL,
            quantization: textEncoderQuantization,
            outputURL: embeddingsURL,
            negOutputURL: negEmbeddingsURL,
            progressHandler: progressHandler
        )

        // Text encoder fully deallocated now - clear cache
        clearGPUMemory()
        await Task.yield()

        // Load embeddings from disk (text encoder memory is now free)
        let promptEmbeds = try MLX.loadArray(url: embeddingsURL)
        let negativePromptEmbeds: MLXArray? = hasNegative ? try MLX.loadArray(url: negEmbeddingsURL) : nil

        // Clean up temp files
        try? FileManager.default.removeItem(at: embeddingsURL)
        try? FileManager.default.removeItem(at: negEmbeddingsURL)

        // 4. Denoise (loads transformer temporarily, saves latents to disk)
        let latentsURL = FileManager.default.temporaryDirectory.appendingPathComponent("latents_\(seed).npy")
        try await denoiseAndSave(
            promptEmbeds: promptEmbeds,
            negativePromptEmbeds: negativePromptEmbeds,
            referenceLatents: referenceLatents,
            patchedHeight: patchedHeight,
            patchedWidth: patchedWidth,
            seqLen: seqLen,
            steps: request.steps,
            seed: seed,
            guidanceScale: Float(request.guidanceScale),
            transformerDirURL: transformerComponent.directoryURL,
            transformerQuantization: transformerQuantization,
            isDistilled: isDistilled,
            outputURL: latentsURL,
            progressHandler: progressHandler
        )

        // Transformer fully deallocated now - clear cache
        clearGPUMemory()
        await Task.yield()

        // Load latents from disk (transformer memory is now free)
        let denoised = try MLX.loadArray(url: latentsURL)
        try? FileManager.default.removeItem(at: latentsURL)

        // 5. Decode and save (loads VAE temporarily)
        let outputURL = try await decodeAndSave(
            latents: denoised,
            patchedHeight: patchedHeight,
            patchedWidth: patchedWidth,
            bnMean: bnMean,
            bnVar: bnVar,
            seed: seed,
            vaeDirURL: vaeComponent.directoryURL,
            requestedOutputURL: request.outputURL,
            progressHandler: progressHandler
        )

        return outputURL
    }

    private func resolveModelPath(from spec: String) throws -> String {
        let fm = FileManager.default
        var isDir: ObjCBool = false

        let localURL = URL(fileURLWithPath: spec).standardizedFileURL
        if fm.fileExists(atPath: localURL.path, isDirectory: &isDir), isDir.boolValue {
            _ = try MereRunModelManifest.loadRequired(from: localURL)
            return localURL.path
        }

        if let id = ModelResolver.ModelID(rawValue: spec) {
            return try ModelResolver().resolve(id).rootURL.path
        }

        throw Flux2Error.modelNotFound(spec)
    }

}
