import Foundation
import MLX

struct InklingLoadedTextModel: Sendable {
    let model: InklingLanguageModel
    let tokenizerAndTemplate: InklingTokenizerAndTemplate
    let config: InklingConfig
    let rootURL: URL
}

enum InklingTextModelLoader {
    static func load(
        modelId: String,
        modelPath: String? = nil,
        maxContextLength: Int? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> InklingLoadedTextModel {
        let rootURL = try await resolveModelRoot(
            modelId: modelId,
            modelPath: modelPath,
            progressHandler: progressHandler
        )
        return try await load(
            rootURL: rootURL,
            maxContextLength: maxContextLength,
            progressHandler: progressHandler
        )
    }

    static func load(
        rootURL: URL,
        maxContextLength: Int? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> InklingLoadedTextModel {
        let normalized = InklingResources.normalizedRootURL(rootURL)
        let missing = InklingResources.validate(rootURL: normalized)
        guard missing.isEmpty else {
            throw InklingError.missingFiles(missing.map(\.path))
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Inkling config"))
        let config = try JSONDecoder().decode(
            InklingConfig.self,
            from: Data(contentsOf: normalized.appendingPathComponent("config.json"))
        )
        guard config.quantization?.bits == InklingResources.quantizationBits,
              config.quantization?.groupSize == InklingResources.quantizationGroupSize,
              config.quantization?.mode == InklingResources.quantizationMode,
              config.quantization?.scope == InklingResources.quantizationScope else {
            throw InklingError.generationFailed(
                "Inkling artifact must use affine 2-bit/group-128 routed experts with BF16 non-routed weights."
            )
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Inkling tokenizer"))
        let tokenizer = try await InklingTokenizerAndTemplate.load(
            from: normalized,
            maxLengthOverride: min(
                maxContextLength ?? InklingResources.defaultContextLength,
                config.textConfig.modelMaxLength
            )
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Inkling weights"))
        let model = InklingLanguageModel(config: config)
        let index = normalized.appendingPathComponent("model.safetensors.index.json")
        let single = normalized.appendingPathComponent("model.safetensors")
        let groupSize = config.quantization?.groupSize ?? InklingResources.quantizationGroupSize
        let bits = config.quantization?.bits ?? InklingResources.quantizationBits
        if FileManager.default.fileExists(atPath: index.path) {
            try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                indexURL: index,
                to: model,
                groupSize: groupSize,
                bits: bits,
                keyMapper: InklingResources.mapWeightKey,
                mapper: InklingResources.mapWeight(key:value:),
                progressHandler: { progress in
                    progressHandler?(ChatProgress(
                        stage: .loadingModel,
                        message: "Loading Inkling shard \(progress.shardIndex + 1)/\(progress.shardCount)"
                    ))
                }
            )
        } else {
            let arrays = try MLX.loadArrays(url: single)
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                arrays,
                to: model,
                groupSize: groupSize,
                bits: bits,
                keyMapper: InklingResources.mapWeightKey,
                mapper: InklingResources.mapWeight(key:value:)
            )
        }
        try Task.checkCancellation()
        return InklingLoadedTextModel(
            model: model,
            tokenizerAndTemplate: tokenizer,
            config: config,
            rootURL: normalized
        )
    }

    static func resolveModelRoot(
        modelId: String,
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        if let modelPath = modelPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !modelPath.isEmpty {
            let rootURL = URL(fileURLWithPath: modelPath).standardizedFileURL
            guard FileManager.default.fileExists(atPath: rootURL.path) else {
                throw InklingError.downloadFailed("Inkling model path does not exist: \(rootURL.path)")
            }
            return rootURL
        }

        let requested = modelId.trimmingCharacters(in: .whitespacesAndNewlines)
        if FileManager.default.fileExists(atPath: requested) {
            return URL(fileURLWithPath: requested).standardizedFileURL
        }
        guard InklingResources.handles(modelSpec: requested) else {
            throw InklingError.unsupportedModelID(requested)
        }
        do {
            let resolution = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: requested,
                defaultModelID: InklingResources.modelID,
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(ChatProgress(
                            stage: .loadingModel,
                            message: "Downloading Inkling... \(percent)%"
                        ))
                    case .extracting:
                        progressHandler?(ChatProgress(
                            stage: .loadingModel,
                            message: "Extracting Inkling..."
                        ))
                    }
                }
            )
            return InklingResources.normalizedRootURL(resolution.url)
        } catch let error as ManagedModelResolver.ResolverError {
            throw InklingError.downloadFailed(error.localizedDescription)
        }
    }
}
