import Foundation
import MLX

struct LagunaLoadedTextModel: Sendable {
    let model: LagunaCausalLM
    let tokenizerAndTemplate: LagunaTokenizerAndTemplate
    let config: LagunaConfig
    let rootURL: URL
}

enum LagunaTextModelLoader {
    static func load(
        modelId: String,
        modelPath: String? = nil,
        maxContextLength: Int? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> LagunaLoadedTextModel {
        let rootURL = try await resolveModelRoot(
            modelId: modelId,
            modelPath: modelPath,
            progressHandler: progressHandler
        )
        let missing = LagunaResources.validate(rootURL: rootURL)
        guard missing.isEmpty else {
            throw LagunaError.missingFiles(missing)
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Laguna config"))
        let config = try JSONDecoder().decode(
            LagunaConfig.self,
            from: Data(contentsOf: rootURL.appending(path: "config.json"))
        )
        let indexURL = rootURL.appending(path: "model.safetensors.index.json")
        let index = try JSONDecoder().decode(
            HFSafetensorsIndex.self,
            from: Data(contentsOf: indexURL)
        )
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Laguna tokenizer"))
        let tokenizer = try await LagunaTokenizerAndTemplate.load(
            from: rootURL,
            maxLength: min(
                maxContextLength ?? LagunaResources.defaultContextLength,
                config.maxPositionEmbeddings
            )
        )
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Laguna weights"))
        let model = LagunaCausalLM(
            config: config,
            quantizedSharedExperts: LagunaResources.hasQuantizedSharedExperts(index)
        )
        try HFSafetensorsWeightsLoader.applyShardedWeights(
            indexURL: indexURL,
            to: model,
            dtype: nil,
            verify: .shapeMismatch,
            progressHandler: { progress in
                progressHandler?(ChatProgress(
                    stage: .loadingModel,
                    message: "Loading Laguna shard \(progress.shardIndex + 1)/\(progress.shardCount)"
                ))
            }
        )
        return LagunaLoadedTextModel(
            model: model,
            tokenizerAndTemplate: tokenizer,
            config: config,
            rootURL: rootURL
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
                throw LagunaError.modelPathRequired
            }
            return rootURL
        }

        let requested = modelId.trimmingCharacters(in: .whitespacesAndNewlines)
        if FileManager.default.fileExists(atPath: requested) {
            return URL(fileURLWithPath: requested).standardizedFileURL
        }
        guard let managedID = LagunaResources.managedModelID(for: requested) else {
            throw LagunaError.modelPathRequired
        }
        if let installed = ManagedModelResolver.resolveInstalledModel(id: managedID) {
            return installed.standardizedFileURL
        }
        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: managedID,
            defaultModelID: managedID,
            progress: { event in
                switch event {
                case .downloading(let percent):
                    progressHandler?(ChatProgress(
                        stage: .loadingModel,
                        message: "Downloading Laguna... \(percent)%"
                    ))
                case .extracting:
                    progressHandler?(ChatProgress(
                        stage: .loadingModel,
                        message: "Extracting Laguna..."
                    ))
                }
            }
        )
        return resolution.url.standardizedFileURL
    }
}
