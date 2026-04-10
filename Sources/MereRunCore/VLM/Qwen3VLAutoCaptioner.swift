import Foundation
import MLX
import MLXNN

/// Auto-captioner using Phi Nano (Apache 2.0 licensed vision-language model).
/// Downloads model on first use via R2, then caches locally.
public actor Qwen3VLAutoCaptioner {
    public static let modelId = "vision-vlm-phi-nano"
    private static let archiveKey = "models/phi-nano.tar.gz"
    private static let archiveSize: Int64 = 1_527_005_575  // ~1.5GB

    public enum CaptionerError: Error, LocalizedError {
        case modelNotReady
        case imageLoadFailed(URL)
        case generationFailed(String)
        case downloadFailed(String)
        case extractionFailed

        public var errorDescription: String? {
            switch self {
            case .modelNotReady: return "VLM model not loaded"
            case .imageLoadFailed(let url): return "Failed to load image: \(url.lastPathComponent)"
            case .generationFailed(let msg): return msg
            case .downloadFailed(let msg): return "Download failed: \(msg)"
            case .extractionFailed: return "Failed to extract model archive"
            }
        }
    }

    public struct DownloadProgress: Sendable {
        public let fraction: Double
        public let status: String
    }

    private var adaptedModelPath: URL?
    private var captioner: QwenVLCaptioner?
    private var isLoading = false

    public init() {}

    /// Check if model is cached
    public func isModelCached() -> Bool {
        cachedModelPath() != nil
    }

    /// Ensure model is downloaded and ready, returns adapted model path
    @discardableResult
    public func ensureReady(
        progressHandler: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        if let path = adaptedModelPath, captioner != nil {
            return path
        }
        if isLoading {
            while isLoading {
                try await Task.sleep(nanoseconds: 100_000_000)
            }
            if let path = adaptedModelPath {
                return path
            }
        }

        isLoading = true
        defer { isLoading = false }

        // Check if already cached
        if let cached = cachedModelPath() {
            progressHandler?(DownloadProgress(fraction: 0.9, status: "Loading model..."))
            adaptedModelPath = cached
            captioner = try QwenVLCaptioner(modelRoot: cached)
            progressHandler?(DownloadProgress(fraction: 1.0, status: "Ready"))
            return cached
        }

        progressHandler?(DownloadProgress(fraction: 0, status: "Downloading Phi Nano..."))

        let path = try await resolveRawModelRoot(progressHandler: progressHandler)

        progressHandler?(DownloadProgress(fraction: 0.9, status: "Loading model..."))

        // Adapt to QwenVLCaptioner format and load
        let adaptedPath = try adaptModelStructure(from: path)
        adaptedModelPath = adaptedPath

        captioner = try QwenVLCaptioner(modelRoot: adaptedPath)

        progressHandler?(DownloadProgress(fraction: 1.0, status: "Ready"))
        return adaptedPath
    }

    /// Caption an image
    public func caption(
        imageURL: URL,
        prompt: String = "Describe this image in one sentence for AI image generation training. Focus on the subject, style, and composition."
    ) async throws -> String {
        guard let captioner = captioner else {
            throw CaptionerError.modelNotReady
        }

        return try captioner.caption(imageURL: imageURL, prompt: prompt)
    }

    // MARK: - Private

    private func cachedModelPath() -> URL? {
        let fm = FileManager.default
        guard let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            return nil
        }

        // Check for adapted structure
        let adaptedPath = appSupport.appendingPathComponent("MereRun/phi-nano")
        if fm.fileExists(atPath: adaptedPath.appendingPathComponent("text_encoder/config.json").path) {
            return adaptedPath
        }

        return nil
    }

    private func resolveRawModelRoot(
        progressHandler: (@Sendable (DownloadProgress) -> Void)?
    ) async throws -> URL {
        do {
            return try await PretrainedModelLoader.fromPretrainedArchive(
                modelPath: nil,
                modelId: Self.modelId,
                defaultModelIds: [Self.modelId],
                storageId: Self.modelId,
                archiveKey: Self.archiveKey,
                archiveSize: Self.archiveSize,
                normalize: { base, fileManager in
                    Self.resolveNestedIfNeeded(base: base, fileManager: fileManager)
                },
                validate: { root, fileManager in
                    Self.validateRawRoot(root, fileManager: fileManager)
                },
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        let clamped = max(0, min(percent, 100))
                        progressHandler?(DownloadProgress(
                            fraction: Double(clamped) / 100.0 * 0.8,
                            status: "Downloading Phi Nano (\(clamped)%)"
                        ))
                    case .extracting:
                        progressHandler?(DownloadProgress(fraction: 0.82, status: "Extracting..."))
                    }
                }
            )
        } catch let error as PretrainedModelLoader.LoadError {
            throw mapModelLoaderError(error)
        }
    }

    private static func resolveNestedIfNeeded(base: URL, fileManager: FileManager) -> URL {
        if validateRawRoot(base, fileManager: fileManager).isEmpty {
            return base
        }

        guard let children = try? fileManager.contentsOfDirectory(
            at: base,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else {
            return base
        }

        let childDirs = children.filter { child in
            var isDirectory: ObjCBool = false
            return fileManager.fileExists(atPath: child.path, isDirectory: &isDirectory)
                && isDirectory.boolValue
        }

        guard childDirs.count == 1 else {
            return base
        }

        let nested = childDirs[0]
        return validateRawRoot(nested, fileManager: fileManager).isEmpty ? nested : base
    }

    private static func validateRawRoot(_ root: URL, fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []

        let configURL = root.appendingPathComponent("config.json")
        if !fileManager.fileExists(atPath: configURL.path) {
            missing.append(configURL)
        }

        let tokenizerJSON = root.appendingPathComponent("tokenizer.json")
        if !fileManager.fileExists(atPath: tokenizerJSON.path) {
            missing.append(tokenizerJSON)
        }

        let tokenizerConfig = root.appendingPathComponent("tokenizer_config.json")
        if !fileManager.fileExists(atPath: tokenizerConfig.path) {
            missing.append(tokenizerConfig)
        }

        let weightURL = root.appendingPathComponent("model.safetensors")
        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        let hasWeights = fileManager.fileExists(atPath: weightURL.path)
            || fileManager.fileExists(atPath: indexURL.path)
        if !hasWeights {
            missing.append(weightURL)
        }

        return missing
    }

    private func mapModelLoaderError(_ error: PretrainedModelLoader.LoadError) -> CaptionerError {
        switch error {
        case .unsupportedModelId(let modelID):
            return .downloadFailed("Unsupported model id: \(modelID)")
        case .missingFiles(let files):
            return .generationFailed("Phi Nano model files missing: \(files.joined(separator: ", "))")
        case .downloadFailed(let message):
            return .downloadFailed(message)
        case .extractionFailed:
            return .extractionFailed
        }
    }

    /// Adapt flat structure to QwenVLCaptioner expected structure
    private func adaptModelStructure(from sourcePath: URL) throws -> URL {
        let fm = FileManager.default
        guard let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            throw CaptionerError.generationFailed("Cannot find app support directory")
        }

        let targetPath = appSupport.appendingPathComponent("MereRun/phi-nano")

        // Check if already adapted
        if fm.fileExists(atPath: targetPath.appendingPathComponent("text_encoder/config.json").path) {
            return targetPath
        }

        // Create directory structure
        let textEncoderDir = targetPath.appendingPathComponent("text_encoder")
        let tokenizerDir = targetPath.appendingPathComponent("tokenizer")

        try fm.createDirectory(at: textEncoderDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: tokenizerDir, withIntermediateDirectories: true)

        // Transform and write config.json for text_encoder
        let sourceConfig = sourcePath.appendingPathComponent("config.json")
        let targetConfig = textEncoderDir.appendingPathComponent("config.json")
        if !fm.fileExists(atPath: targetConfig.path) {
            try transformConfig(from: sourceConfig, to: targetConfig)
        }

        // Copy/link model weights
        let sourceWeights = sourcePath.appendingPathComponent("model.safetensors")
        let targetWeights = textEncoderDir.appendingPathComponent("model.safetensors")
        if fm.fileExists(atPath: sourceWeights.path) && !fm.fileExists(atPath: targetWeights.path) {
            try fm.createSymbolicLink(at: targetWeights, withDestinationURL: sourceWeights)
        }

        // Handle sharded weights
        let indexFile = sourcePath.appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: indexFile.path) {
            let targetIndex = textEncoderDir.appendingPathComponent("model.safetensors.index.json")
            if !fm.fileExists(atPath: targetIndex.path) {
                try fm.createSymbolicLink(at: targetIndex, withDestinationURL: indexFile)
            }

            // Link shard files
            let contents = try fm.contentsOfDirectory(at: sourcePath, includingPropertiesForKeys: nil)
            for file in contents where file.lastPathComponent.hasPrefix("model-") && file.pathExtension == "safetensors" {
                let targetShard = textEncoderDir.appendingPathComponent(file.lastPathComponent)
                if !fm.fileExists(atPath: targetShard.path) {
                    try fm.createSymbolicLink(at: targetShard, withDestinationURL: file)
                }
            }
        }

        // Symlink model weights into vision_tower so vision weights are loaded too
        // (the single safetensors file contains both text and vision weights)
        let visionDir = targetPath.appendingPathComponent("vision_tower")
        try fm.createDirectory(at: visionDir, withIntermediateDirectories: true)
        let visionWeightsTarget = visionDir.appendingPathComponent("model.safetensors")
        if !fm.fileExists(atPath: visionWeightsTarget.path) {
            let sourceWeights = sourcePath.appendingPathComponent("model.safetensors")
            if fm.fileExists(atPath: sourceWeights.path) {
                try fm.createSymbolicLink(at: visionWeightsTarget, withDestinationURL: sourceWeights)
            }
        }
        let visionIndexTarget = visionDir.appendingPathComponent("model.safetensors.index.json")
        if !fm.fileExists(atPath: visionIndexTarget.path) {
            let sourceIndex = sourcePath.appendingPathComponent("model.safetensors.index.json")
            if fm.fileExists(atPath: sourceIndex.path) {
                try fm.createSymbolicLink(at: visionIndexTarget, withDestinationURL: sourceIndex)
            }
        }

        // Copy tokenizer files
        let tokenizerFiles = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json"]
        for filename in tokenizerFiles {
            let source = sourcePath.appendingPathComponent(filename)
            let target = tokenizerDir.appendingPathComponent(filename)
            if fm.fileExists(atPath: source.path) && !fm.fileExists(atPath: target.path) {
                try fm.createSymbolicLink(at: target, withDestinationURL: source)
            }
        }

        return targetPath
    }

    /// Transform config.json to flatten text_config to root level
    private func transformConfig(from source: URL, to target: URL) throws {
        let data = try Data(contentsOf: source)
        guard var json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CaptionerError.generationFailed("Invalid config.json format")
        }

        // If text_config exists, merge its contents to root level
        if let textConfig = json["text_config"] as? [String: Any] {
            for (key, value) in textConfig {
                if json[key] == nil {
                    json[key] = value
                }
            }
        }

        let outputData = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
        try outputData.write(to: target)
    }
}
