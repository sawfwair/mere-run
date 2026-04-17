import Foundation
import MLX

/// Owns model resolution, resource wiring, and weight loading.
/// This file intentionally stops before prompt encoding or latent/image work.
extension ZImageTurboGenerator {
    func loadModelIfNeeded(
        modelSpec: String,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> LoadedModel {
        if let loaded, loaded.modelSpec == modelSpec {
            return loaded
        }

        progressHandler?(GenerationProgress(stage: .loadingModel, stepIndex: 0, totalSteps: 1))

        let resolved = try await resolveModelRoot(modelSpec: modelSpec, progressHandler: progressHandler)
        let manifest = try MereRunModelManifest.loadRequired(from: resolved)
        let componentResolver = ModelComponentResolver(modelRootURL: resolved, manifest: manifest)
        let tokenizerComponent = try componentResolver.resolveDirectory(for: .tokenizer, fallbackLocalPath: "tokenizer")
        let textEncoderComponent = try componentResolver.resolveDirectory(for: .textEncoder, fallbackLocalPath: "text_encoder")
        let transformerComponent = try componentResolver.resolveDirectory(for: .transformer, fallbackLocalPath: "transformer")
        let vaeComponent = try componentResolver.resolveDirectory(for: .vae, fallbackLocalPath: "vae")
        let schedulerComponent = try componentResolver.resolveDirectory(for: .scheduler, fallbackLocalPath: "scheduler")

        let resources = ZImageTurboResources(
            modelRootURL: resolved,
            tokenizerDirURL: tokenizerComponent.directoryURL,
            textEncoderDirURL: textEncoderComponent.directoryURL,
            transformerDirURL: transformerComponent.directoryURL,
            vaeDirURL: vaeComponent.directoryURL,
            schedulerDirURL: schedulerComponent.directoryURL
        )
        let missing = resources.validate()
        if !missing.isEmpty {
            throw ZImageTurboModelContainer.ContainerError.missingModelFiles(missing)
        }

        let configs = try ZImageTurboModelConfigs.load(from: resources)
        let textEncoderQuantization = try ModelWeightsLoader.QuantizationParams.fromManifest(textEncoderComponent.sourceManifest)
        let transformerQuantization = try ModelWeightsLoader.QuantizationParams.fromManifest(transformerComponent.sourceManifest)

        let tokenizerDir = resources.tokenizerDirURL
        guard FileManager.default.fileExists(atPath: tokenizerDir.path) else {
            throw GeneratorError.tokenizerMissing(tokenizerDir)
        }
        let tokenizer = try QwenTokenizer.load(from: tokenizerDir, maxLengthOverride: configs.textEncoder.maxPositionEmbeddings)

        let textEncoder = QwenTextEncoder(configuration: .init(
            vocabSize: configs.textEncoder.vocabSize,
            hiddenSize: configs.textEncoder.hiddenSize,
            numHiddenLayers: configs.textEncoder.numHiddenLayers,
            numAttentionHeads: configs.textEncoder.numAttentionHeads,
            numKeyValueHeads: configs.textEncoder.numKeyValueHeads,
            intermediateSize: configs.textEncoder.intermediateSize,
            ropeTheta: configs.textEncoder.ropeTheta,
            maxPositionEmbeddings: configs.textEncoder.maxPositionEmbeddings,
            rmsNormEps: configs.textEncoder.rmsNormEps,
            headDim: configs.textEncoder.headDim
        ))
        let transformer = ZImageTransformer2DModel(configuration: configs.transformer)
        let vae = AutoencoderKL(configuration: .init(
            inChannels: configs.vae.inChannels,
            outChannels: configs.vae.outChannels,
            latentChannels: configs.vae.latentChannels,
            scalingFactor: configs.vae.scalingFactor,
            shiftFactor: configs.vae.shiftFactor,
            blockOutChannels: configs.vae.blockOutChannels,
            layersPerBlock: configs.vae.layersPerBlock,
            normNumGroups: configs.vae.normNumGroups,
            sampleSize: configs.vae.sampleSize ?? ZImageModelMetadata.VAE.sampleSize,
            midBlockAddAttention: configs.vae.midBlockAddAttention
        ))

        try loadTextEncoderWeights(
            resources: resources,
            into: textEncoder,
            quantization: textEncoderQuantization,
            progressHandler: progressHandler
        )
        try loadTransformerWeights(
            resources: resources,
            into: transformer,
            quantization: transformerQuantization,
            progressHandler: progressHandler
        )
        try loadVAEWeights(resources: resources, into: vae, progressHandler: progressHandler)

        let loadedModel = LoadedModel(
            modelSpec: modelSpec,
            rootURL: resolved,
            manifest: manifest,
            resources: resources,
            configs: configs,
            textEncoderQuantization: textEncoderQuantization,
            transformerQuantization: transformerQuantization,
            tokenizer: tokenizer,
            textEncoder: textEncoder,
            transformer: transformer,
            vae: vae
        )

        loaded = loadedModel
        return loadedModel
    }

    func resolveModelRoot(
        modelSpec: String,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> URL {
        let local = URL(fileURLWithPath: modelSpec).standardizedFileURL
        if FileManager.default.fileExists(atPath: local.path) {
            _ = try MereRunModelManifest.loadRequired(from: local)
            return local
        }

        return try await ZImageTurboRepository.resolveRemoteModelRoot(
            modelSpec: modelSpec,
            progress: { event in
                guard let progressHandler else { return }
                switch event {
                case .downloading(let percent):
                    progressHandler(GenerationProgress(stage: .loadingModel, stepIndex: percent, totalSteps: 100))
                case .extracting:
                    progressHandler(GenerationProgress(stage: .loadingModel, stepIndex: 100, totalSteps: 100))
                }
            }
        )
    }

    func loadTextEncoderWeights(
        resources: ZImageTurboResources,
        into model: QwenTextEncoder,
        quantization: ModelWeightsLoader.QuantizationParams?,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws {
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.textEncoderWeightsIndexURL,
            singleURL: resources.textEncoderWeightsURL,
            to: model,
            dtype: .bfloat16,
            verify: [.noUnusedKeys, .shapeMismatch],
            mapper: { key, value in
                if key.hasPrefix("model.") {
                    let remainder = String(key.dropFirst("model.".count))
                    return [("encoder.\(remainder)", value)]
                }
                return [(key, value)]
            },
            keyMapper: { key in
                if key.hasPrefix("model.") {
                    return "encoder." + String(key.dropFirst("model.".count))
                }
                if key.hasPrefix("encoder.") {
                    return key
                }
                return "encoder.\(key)"
            },
            quantization: quantization,
            progressHandler: { shard in
                progressHandler?(GenerationProgress(
                    stage: .encodingText,
                    stepIndex: shard.shardIndex + 1,
                    totalSteps: shard.shardCount
                ))
            }
        )
    }

    func loadTransformerWeights(
        resources: ZImageTurboResources,
        into model: ZImageTransformer2DModel,
        quantization: ModelWeightsLoader.QuantizationParams?,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws {
        let transformerKeyMapper: (String) -> String = { key in
            var mapped = key
            if key.contains("t_embedder.linear1") {
                mapped = key.replacingOccurrences(of: "t_embedder.linear1", with: "t_embedder.mlp.0")
            } else if key.contains("t_embedder.linear2") {
                mapped = key.replacingOccurrences(of: "t_embedder.linear2", with: "t_embedder.mlp.2")
            } else if key.contains("all_final_layer") && key.contains("adaLN_modulation.0.") {
                mapped = key.replacingOccurrences(of: "adaLN_modulation.0.", with: "adaLN_modulation.1.")
            }
            return mapped
        }

        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.transformerWeightsIndexURL,
            singleURL: resources.transformerWeightsURL,
            to: model,
            dtype: .bfloat16,
            verify: [.noUnusedKeys, .shapeMismatch],
            keyMapper: transformerKeyMapper,
            quantization: quantization,
            progressHandler: { shard in
                progressHandler?(GenerationProgress(
                    stage: .loadingTransformer,
                    stepIndex: shard.shardIndex + 1,
                    totalSteps: shard.shardCount
                ))
            }
        )
    }

    func loadVAEWeights(
        resources: ZImageTurboResources,
        into model: AutoencoderKL,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws {
        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))

        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.vaeWeightsURL,
            to: model,
            dtype: .bfloat16,
            verify: [.noUnusedKeys, .shapeMismatch],
            mapper: { key, value in
                let maybeConverted = value.ndim == 4 ? HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(value) : value
                return [(key, maybeConverted)]
            }
        )
    }
}
