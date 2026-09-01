import Foundation
import MLX

public struct Gemma4LoadedUnifiedModel: Sendable {
    public let model: Gemma4UnifiedCausalLM
    public let tokenizerAndTemplate: Gemma4TokenizerAndTemplate
    public let config: Gemma4Config
    public let rootURL: URL

    public init(
        model: Gemma4UnifiedCausalLM,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        config: Gemma4Config,
        rootURL: URL
    ) {
        self.model = model
        self.tokenizerAndTemplate = tokenizerAndTemplate
        self.config = config
        self.rootURL = rootURL
    }
}

public enum Gemma4UnifiedModelLoader {
    public static func load(
        modelId: String,
        modelPath: String? = nil,
        maxContextLength: Int? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> Gemma4LoadedUnifiedModel {
        guard Gemma4Resources.supportsVision(modelSpec: modelId) else {
            throw Gemma4Error.unsupportedConfiguration(
                "Native Gemma4 VLM training requires \(Gemma4Resources.visionTwelveBModelId)."
            )
        }

        let rootURL = try await Gemma4TextModelLoader.resolveModelRoot(
            modelId: modelId,
            modelPath: modelPath,
            progressHandler: progressHandler
        )
        let normalizedRoot = Gemma4Resources.normalizedRootURL(rootURL)
        let resources = Gemma4Resources(rootURL: normalizedRoot)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw Gemma4Error.missingFiles(missing.map(\.lastPathComponent))
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Gemma4 unified config"))
        let configData = try Data(contentsOf: resources.configURL)
        let config = try JSONDecoder().decode(Gemma4Config.self, from: configData)
        guard config.visionConfig != nil,
              config.imageTokenId != nil,
              config.boiTokenId != nil,
              config.eoiTokenId != nil else {
            throw Gemma4Error.unsupportedConfiguration(
                "Gemma4 VLM training requires unified vision and image-token configuration."
            )
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Gemma4 tokenizer"))
        let tokenizer = try await Gemma4TokenizerAndTemplate.load(
            from: normalizedRoot,
            maxLengthOverride: min(
                maxContextLength ?? Gemma4Resources.defaultContextLength,
                config.textConfig.maxPositionEmbeddings
            )
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Gemma4 unified weights"))
        let model = try Gemma4UnifiedCausalLM(config: config)
        try loadWeights(into: model, from: resources)
        return Gemma4LoadedUnifiedModel(
            model: model,
            tokenizerAndTemplate: tokenizer,
            config: config,
            rootURL: normalizedRoot
        )
    }

    static func loadWeights(
        into model: Gemma4UnifiedCausalLM,
        from resources: Gemma4Resources
    ) throws {
        let include: (String) -> Bool = { key in
            normalizedWeightKey(key) != nil
        }
        let mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            guard let mapped = normalizedWeightKey(key) else {
                return []
            }
            return [(mapped, value)]
        }

        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            try HFSafetensorsWeightsLoader.applyShardedWeights(
                indexURL: resources.modelIndexURL,
                to: model,
                dtype: .bfloat16,
                verify: .none,
                mapper: mapper
            )
        } else if FileManager.default.fileExists(atPath: resources.modelWeightsURL.path) {
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: resources.modelWeightsURL,
                to: model,
                dtype: .bfloat16,
                verify: .none,
                include: include,
                mapper: mapper,
                batchSize: 24
            )
        } else {
            throw Gemma4Error.missingFiles([
                resources.modelIndexURL.lastPathComponent,
                resources.modelWeightsURL.lastPathComponent,
            ])
        }
    }

    static func normalizedWeightKey(_ key: String) -> String? {
        guard !key.contains("rotary_emb"),
              key != "lm_head.weight",
              !key.contains("embed_audio") else {
            return nil
        }

        let withoutModelPrefix = key.hasPrefix("model.")
            ? String(key.dropFirst("model.".count))
            : key
        if withoutModelPrefix.hasPrefix("language_model.model.") {
            return "language_model.\(withoutModelPrefix.dropFirst("language_model.model.".count))"
        }
        if withoutModelPrefix.hasPrefix("language_model.")
            || withoutModelPrefix.hasPrefix("vision_embedder.")
            || withoutModelPrefix.hasPrefix("embed_vision.") {
            return withoutModelPrefix
        }
        return nil
    }
}
