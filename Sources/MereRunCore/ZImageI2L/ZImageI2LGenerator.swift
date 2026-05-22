import Foundation
import MediaIO
import MLX
import MLXNN
import MLXRandom

// MARK: - Z-Image-i2L Generator
//
// Converts input images to LoRA weights using:
// 1. SigLIP2-G384 vision encoder (1536-dim)
// 2. DINOv3-7B vision encoder (4096-dim)
// 3. Z-Image-i2L projection model
//
// Pipeline:
//   images -> [SigLIP2 || DINOv3] -> concat(5632-dim) -> I2L model -> LoRA weights

public actor ZImageI2LGenerator {
    private struct LoadedModels {
        let siglip2: SigLIP2VisionModel
        let dinov3: DINOv3VisionModel
        let i2l: ZImageI2LModel
        let configs: ZImageI2LModelConfigs
    }

    private var loaded: LoadedModels?
    private let dtype: DType = .bfloat16

    /// Optional explicit model paths (for app-bundled models)
    private var explicitSiglip2Path: URL?
    private var explicitDinov3Path: URL?
    private var explicitI2LPath: URL?
    private var resolvedModelRoot: URL?

    public init() {}

    /// Initialize with explicit model paths (for app-bundled models)
    public init(siglip2Path: URL? = nil, dinov3Path: URL? = nil, i2lPath: URL? = nil) {
        self.explicitSiglip2Path = siglip2Path
        self.explicitDinov3Path = dinov3Path
        self.explicitI2LPath = i2lPath
    }

    // MARK: - Public API

    /// Generate LoRA weights from input images
    public func generateLoRA(
        from images: [URL],
        outputPath: URL,
        progressHandler: (@Sendable (String) -> Void)? = nil
    ) async throws {
        progressHandler?("Loading models...")
        let models = try await loadModelsIfNeeded(progressHandler: progressHandler)

        progressHandler?("Processing \(images.count) image(s)...")

        // Load and preprocess images
        var combinedEmbeddings: [MLXArray] = []
        combinedEmbeddings.reserveCapacity(images.count)

        for (idx, imageURL) in images.enumerated() {
            progressHandler?("Encoding image \(idx + 1)/\(images.count)...")

            // Match DiffSynth preprocessing:
            // - center crop + resize to 1024x1024 (aspect-fill)
            // - SigLIP2 processor then resizes to 384 and normalizes with mean/std = 0.5
            // - DINOv3 processor then resizes to 224 and normalizes with ImageNet stats
            let (siglipInput, dinoInput) = try loadImageForEncoders(from: imageURL)

            // Encode with SigLIP2 (384x384)
            let siglipEmb = models.siglip2.encode(siglipInput)
            // Encode with DINOv3 (224x224)
            let dinoEmb = models.dinov3.encode(dinoInput)

            // Ensure embeddings are 1D, then concatenate.
            let siglip1D = siglipEmb.ndim == 2 ? siglipEmb.squeezed(axis: 0) : siglipEmb
            let dino1D = dinoEmb.ndim == 2 ? dinoEmb.squeezed(axis: 0) : dinoEmb
            let combined = MLX.concatenated([siglip1D, dino1D], axis: -1)  // [5632]
            combinedEmbeddings.append(combined)

            eval(siglipEmb, dinoEmb, combined)
        }

        progressHandler?("Generating LoRA weights...")
        var perImageLoRAs: [[String: MLXArray]] = []
        perImageLoRAs.reserveCapacity(combinedEmbeddings.count)
        for embedding in combinedEmbeddings {
            let loraOutput = models.i2l(embedding)
            let loraDict = loraOutput.toDict()
            // Evaluate the LoRA arrays
            eval(Array(loraDict.values))
            perImageLoRAs.append(loraDict)
        }

        // Merge LoRAs across images (DiffSynth `merge_lora`):
        // - concat lora_A along dim 0
        // - concat lora_B along dim 1
        // - scale lora_A by alpha=1/N
        let merged: [String: MLXArray] = {
            guard let first = perImageLoRAs.first else { return [:] }
            guard perImageLoRAs.count > 1 else { return first }

            let alpha = Float(1.0) / Float(perImageLoRAs.count)
            let alphaArr = MLXArray(alpha)

            var out: [String: MLXArray] = [:]
            out.reserveCapacity(first.count)

            for key in first.keys where key.contains(".lora_A.") {
                let upKey = key.replacingOccurrences(of: ".lora_A.", with: ".lora_B.")
                guard first[upKey] != nil else { continue }

                var downs: [MLXArray] = []
                var ups: [MLXArray] = []
                downs.reserveCapacity(perImageLoRAs.count)
                ups.reserveCapacity(perImageLoRAs.count)

                for lora in perImageLoRAs {
                    guard let down = lora[key], let up = lora[upKey] else { continue }
                    downs.append(down)
                    ups.append(up)
                }

                // Defensive: if any image is missing a key, skip merging that key.
                guard downs.count == perImageLoRAs.count, ups.count == perImageLoRAs.count else { continue }

                let mergedDown = MLX.concatenated(downs, axis: 0) * alphaArr
                let mergedUp = MLX.concatenated(ups, axis: 1)
                out[key] = mergedDown
                out[upKey] = mergedUp
            }

            return out
        }()

        progressHandler?("Saving to \(outputPath.lastPathComponent)...")

        // Evaluate all arrays before saving (MLX lazy evaluation)
        eval(Array(merged.values))

        try saveLoRA(merged, to: outputPath)

        progressHandler?("Done! LoRA saved to \(outputPath.path)")
    }

    /// Unload models to free memory
    public func unload() {
        loaded = nil
        Memory.clearCache()
    }

    // MARK: - Model Loading

    private func loadModelsIfNeeded(progressHandler: (@Sendable (String) -> Void)?) async throws -> LoadedModels {
        if let loaded = loaded {
            return loaded
        }

        let configs = ZImageI2LModelConfigs()

        progressHandler?("Loading SigLIP2-G384...")
        let siglip2 = try await loadSigLIP2(config: configs.siglip2, progressHandler: progressHandler)

        progressHandler?("Loading DINOv3-7B...")
        let dinov3 = try await loadDINOv3(config: configs.dinov3, progressHandler: progressHandler)

        progressHandler?("Loading Z-Image-i2L...")
        let i2l = try await loadI2L(config: configs.i2l, progressHandler: progressHandler)

        let models = LoadedModels(
            siglip2: siglip2,
            dinov3: dinov3,
            i2l: i2l,
            configs: configs
        )

        loaded = models
        return models
    }

    private func loadSigLIP2(config: SigLIP2Config, progressHandler: (@Sendable (String) -> Void)?) async throws -> SigLIP2VisionModel {
        var weightsURL: URL?

        // Check explicit path first (for app-bundled models)
        if let explicit = explicitSiglip2Path, FileManager.default.fileExists(atPath: explicit.path) {
            weightsURL = explicit
        }

        if weightsURL == nil {
            let modelRoot = try await resolveModelRoot(progressHandler: progressHandler)
            let resources = ZImageI2LResources(rootURL: modelRoot)
            weightsURL = resources.resolvedSigLIP2WeightsURL()
        }

        guard let weightsURL = weightsURL else {
            throw ZImageI2LError.modelNotFound(
                "SigLIP2-G384 not found in local i2l model store. Run `mere.run model pull zeta-i2l`."
            )
        }

        let model = SigLIP2VisionModel(config: config)

        progressHandler?("Loading SigLIP2 weights...")
        let rawWeights = try MLX.loadArrays(url: weightsURL)
        // Transpose conv weights from PyTorch OIHW to MLX OHWI format
        var weights: [String: MLXArray] = [:]
        for (key, value) in rawWeights {
            if value.ndim == 4 {
                weights[key] = HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(value)
            } else {
                weights[key] = value
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)

        eval(model)
        return model
    }

    private func loadDINOv3(config: DINOv3Config, progressHandler: (@Sendable (String) -> Void)?) async throws -> DINOv3VisionModel {
        var weightsURL: URL?

        // Check explicit path first (for app-bundled models)
        if let explicit = explicitDinov3Path, FileManager.default.fileExists(atPath: explicit.path) {
            weightsURL = explicit
        }

        if weightsURL == nil {
            let modelRoot = try await resolveModelRoot(progressHandler: progressHandler)
            let resources = ZImageI2LResources(rootURL: modelRoot)
            weightsURL = resources.resolvedDINOv3WeightsURL()
        }

        guard let weightsURL = weightsURL else {
            throw ZImageI2LError.modelNotFound(
                "DINOv3-7B not found in local i2l model store. Run `mere.run model pull zeta-i2l`."
            )
        }

        let model = DINOv3VisionModel(config: config)

        progressHandler?("Loading DINOv3 weights...")
        let rawWeights = try MLX.loadArrays(url: weightsURL)
        // Transpose conv weights from PyTorch OIHW to MLX OHWI format
        var weights: [String: MLXArray] = [:]
        for (key, value) in rawWeights {
            if value.ndim == 4 {
                weights[key] = HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(value)
            } else {
                weights[key] = value
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)

        eval(model)
        return model
    }

    private func loadI2L(config: ZImageI2LConfig, progressHandler: (@Sendable (String) -> Void)?) async throws -> ZImageI2LModel {
        progressHandler?("Loading Z-Image-i2L model...")

        var weightsURL: URL?

        // Check explicit path first (for app-bundled models)
        if let explicit = explicitI2LPath, FileManager.default.fileExists(atPath: explicit.path) {
            weightsURL = explicit
        }

        if weightsURL == nil {
            let modelRoot = try await resolveModelRoot(progressHandler: progressHandler)
            let resources = ZImageI2LResources(rootURL: modelRoot)
            weightsURL = resources.resolvedI2LModelURL()
        }

        guard let weightsURL = weightsURL else {
            throw ZImageI2LError.modelNotFound(
                "Z-Image-i2L model not found in local i2l model store. Run `mere.run model pull zeta-i2l`."
            )
        }

        let model = ZImageI2LModel(config: config)

        let weights = try MLX.loadArrays(url: weightsURL)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)

        eval(model)
        return model
    }

    private func resolveModelRoot(
        progressHandler: (@Sendable (String) -> Void)?
    ) async throws -> URL {
        if let resolvedModelRoot {
            return resolvedModelRoot
        }

        if let installed = ZImageI2LResources.resolveInstalledRoot(fileManager: .default) {
            resolvedModelRoot = installed
            return installed
        }

        let root = try await prepareHuggingFaceModelRoot(progressHandler: progressHandler)
        resolvedModelRoot = root
        return root
    }

    private func prepareHuggingFaceModelRoot(
        progressHandler: (@Sendable (String) -> Void)?
    ) async throws -> URL {
        let fileManager = FileManager.default
        let modelDir = MereRunModelPaths.resolveModelDir(ZImageI2LRepository.modelId) { root in
            let resolved = ZImageI2LResources.resolveNestedIfNeeded(base: root, fileManager: fileManager)
            return ZImageI2LResources(rootURL: resolved).validate(fileManager: fileManager).isEmpty
        }
        try fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let i2lSnapshot = try await prepareHubSnapshot(
            config: ZImageI2LRepository.hubFallbackConfig,
            label: "Z-Image-i2L",
            progressHandler: progressHandler
        )
        let encoderSnapshot = try await prepareHubSnapshot(
            config: GeneralImageEncodersRepository.hubFallbackConfig,
            label: "image encoders",
            progressHandler: progressHandler
        )

        try createOrReplaceSymlink(
            at: modelDir.appendingPathComponent("z-image-i2l", isDirectory: true),
            to: i2lSnapshot,
            fileManager: fileManager
        )
        try createOrReplaceSymlink(
            at: modelDir.appendingPathComponent("general-image-encoders", isDirectory: true),
            to: encoderSnapshot,
            fileManager: fileManager
        )

        let resolved = ZImageI2LResources.resolveNestedIfNeeded(base: modelDir, fileManager: fileManager)
        let missing = ZImageI2LResources(rootURL: resolved).validate(fileManager: fileManager)
        guard missing.isEmpty else {
            throw ZImageI2LError.modelNotFound(
                "Downloaded image-to-LoRA model is incomplete: \(missing.map(\.lastPathComponent).joined(separator: ", "))"
            )
        }
        return resolved
    }

    private func prepareHubSnapshot(
        config: HubFallbackConfig,
        label: String,
        progressHandler: (@Sendable (String) -> Void)?
    ) async throws -> URL {
        let snapshot = try HubSnapshot(
            options: HubSnapshotOptions(
                repoId: config.repoId,
                revision: config.revision,
                patterns: config.patterns
            )
        )
        return try await snapshot.prepare { progress in
            let percent = min(100, max(0, Int(progress.fractionCompleted * 100)))
            progressHandler?("Downloading \(label)... \(percent)%")
        }
    }

    private func createOrReplaceSymlink(
        at linkURL: URL,
        to targetURL: URL,
        fileManager: FileManager
    ) throws {
        if let resourceValues = try? linkURL.resourceValues(forKeys: [.isSymbolicLinkKey]),
           resourceValues.isSymbolicLink == true {
            try fileManager.removeItem(at: linkURL)
        } else if fileManager.fileExists(atPath: linkURL.path) {
            return
        }
        try fileManager.createSymbolicLink(at: linkURL, withDestinationURL: targetURL)
    }

    // MARK: - Image Processing

    private func loadImageForEncoders(from url: URL) throws -> (siglip: MLXArray, dino: MLXArray) {
        let image: MediaImage
        do {
            image = try MediaImageIO.decode(url)
        } catch {
            throw ZImageI2LError.imageLoadFailed(url.path)
        }

        // 1) Crop+resize to 1024x1024 using DiffSynth's `ImageCropAndResize` logic.
        let highRes = try MediaImageIO.centerCropped(image, width: 1024, height: 1024)

        // 2) Resize for each encoder.
        let siglipImage = try MediaImageIO.resized(highRes, width: 384, height: 384)
        let dinoImage = try MediaImageIO.resized(highRes, width: 224, height: 224)

        // 3) Convert to NCHW tensors with the correct per-encoder normalization.
        let siglip = try imageToMLXArray(
            siglipImage,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5]
        )
        let dino = try imageToMLXArray(
            dinoImage,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225]
        )
        return (siglip: siglip, dino: dino)
    }

    private func imageToMLXArray(_ image: MediaImage, mean: [Float], std: [Float]) throws -> MLXArray {
        guard mean.count == 3, std.count == 3 else {
            throw ZImageI2LError.imageProcessingFailed
        }

        let width = image.width
        let height = image.height
        var rgbData: [Float] = []
        rgbData.reserveCapacity(width * height * 3)

        for channel in 0..<3 {
            for pixel in 0..<(width * height) {
                let value = Float(image.rgba8[pixel * 4 + channel]) / 255.0
                rgbData.append((value - mean[channel]) / std[channel])
            }
        }

        return MLXArray(rgbData).reshaped(1, 3, height, width).asType(dtype)
    }

    // MARK: - LoRA Saving

    private func saveLoRA(_ weights: [String: MLXArray], to url: URL) throws {
        // Ensure directory exists
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        // Save as safetensors
        try MLX.save(arrays: weights, url: url)
    }
}

// MARK: - Errors

public enum ZImageI2LError: Error, LocalizedError {
    case modelNotFound(String)
    case imageLoadFailed(String)
    case imageProcessingFailed
    case unsupportedPlatform

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let msg): return msg
        case .imageLoadFailed(let path): return "Failed to load image: \(path)"
        case .imageProcessingFailed: return "Failed to process image"
        case .unsupportedPlatform: return "Platform not supported"
        }
    }
}
