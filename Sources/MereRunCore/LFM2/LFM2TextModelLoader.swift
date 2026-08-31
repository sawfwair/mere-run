import Foundation
import MLX

struct LFM2LoadedTextModel: Sendable {
    let model: LFM2Model
    let tokenizerAndTemplate: LFM2TokenizerAndTemplate
    let config: LFM2Config
    let rootURL: URL
}

enum LFM2TextModelLoader {
    static func load(
        modelId: String,
        modelPath: String? = nil,
        maxContextLength: Int? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> LFM2LoadedTextModel {
        guard LFM2Resources.supportsTextLoRATraining(modelSpec: modelId) else {
            throw LFM2Error.unsupportedModelId(modelId)
        }
        let rootURL = try await resolveModelRoot(
            modelId: modelId,
            modelPath: modelPath,
            progressHandler: progressHandler
        )
        let normalized = LFM2Resources.normalizedRootURL(rootURL)
        let resources = LFM2Resources(rootURL: normalized)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw LFM2Error.missingFiles(missing.map(\.lastPathComponent))
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading LFM2 config"))
        let config = try JSONDecoder().decode(
            LFM2Config.self,
            from: Data(contentsOf: resources.configURL)
        )
        guard config.modelType == "lfm2_moe",
              config.quantization?.bits == 8,
              config.quantization?.mode == "affine" else {
            throw LFM2Error.generationFailed(
                "Native LFM2 text LoRA training v1 requires the affine 8-bit LFM2.5 A1B checkpoint."
            )
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading LFM2 tokenizer"))
        let tokenizer = try await LFM2TokenizerAndTemplate.load(
            from: normalized,
            maxLengthOverride: min(
                maxContextLength ?? LFM2Resources.defaultContextLength,
                config.maxPositionEmbeddings
            )
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading LFM2 weights"))
        let model = LFM2Model(config: config)
        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 8
        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                indexURL: resources.modelIndexURL,
                to: model,
                groupSize: groupSize,
                bits: bits,
                mapper: LFM2Resources.mapWeight(key:value:),
                progressHandler: { progress in
                    progressHandler?(ChatProgress(
                        stage: .loadingModel,
                        message: "Loading LFM2 shard \(progress.shardIndex + 1)/\(progress.shardCount)"
                    ))
                }
            )
        } else {
            let arrays = try MLX.loadArrays(url: resources.modelWeightsURL)
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                arrays,
                to: model,
                groupSize: groupSize,
                bits: bits,
                mapper: LFM2Resources.mapWeight(key:value:)
            )
        }
        try Task.checkCancellation()
        return LFM2LoadedTextModel(
            model: model,
            tokenizerAndTemplate: tokenizer,
            config: config,
            rootURL: normalized
        )
    }

    private static func resolveModelRoot(
        modelId: String,
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        if let explicit = modelPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !explicit.isEmpty {
            let rootURL = URL(fileURLWithPath: explicit).standardizedFileURL
            guard FileManager.default.fileExists(atPath: rootURL.path) else {
                throw LFM2Error.downloadFailed("LFM2 model path does not exist: \(rootURL.path)")
            }
            return rootURL
        }

        let requested = modelId.trimmingCharacters(in: .whitespacesAndNewlines)
        if FileManager.default.fileExists(atPath: requested) {
            return URL(fileURLWithPath: requested).standardizedFileURL
        }
        if let installed = ManagedModelResolver.resolveInstalledModel(id: LFM2Resources.defaultModelId) {
            return installed.standardizedFileURL
        }
        do {
            let resolution = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: LFM2Resources.defaultModelId,
                defaultModelID: LFM2Resources.defaultModelId,
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(ChatProgress(
                            stage: .loadingModel,
                            message: "Downloading LFM2... \(percent)%"
                        ))
                    case .extracting:
                        progressHandler?(ChatProgress(
                            stage: .loadingModel,
                            message: "Extracting LFM2..."
                        ))
                    }
                }
            )
            return resolution.url.standardizedFileURL
        } catch let error as ManagedModelResolver.ResolverError {
            throw LFM2Error.downloadFailed(error.localizedDescription)
        }
    }
}
