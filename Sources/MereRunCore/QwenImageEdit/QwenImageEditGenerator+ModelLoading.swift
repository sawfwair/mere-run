import Foundation
import MLX
import MLXNN

/// Owns model resolution, optional quantization wiring, and weight mapping.
/// This file does not perform inference; it only prepares the edit runtime.
extension QwenImageEditGenerator {
    func loadBaseModelIfNeeded(
        modelSpec: String,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> LoadedModel {
        if let loaded, loaded.modelSpec == modelSpec {
            return loaded
        }

        progressHandler?(GenerationProgress(stage: .loadingModel, stepIndex: 0, totalSteps: 1))

        let resolved = try await resolveModelRoot(modelSpec: modelSpec, progressHandler: progressHandler)
        let resources = QwenImageEditResources(rootURL: resolved)
        let missing = resources.validate()
        if !missing.isEmpty {
            throw QwenImageEditModelContainer.ContainerError.missingModelFiles(missing)
        }

        let configs = try QwenImageEditModelConfigs.load(from: resources)
        let tokenizerDir = resolved.appending(path: "tokenizer", directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: tokenizerDir.path) else {
            throw GeneratorError.tokenizerMissing(tokenizerDir)
        }
        let tokenizer = try Qwen25VLTokenizer.load(from: tokenizerDir)

        let quantConfigURL = resolved.appendingPathComponent("quantization_config.json")
        let quantConfig = FileManager.default.fileExists(atPath: quantConfigURL.path)
            ? try QuantizedWeightLoader.loadConfig(from: quantConfigURL)
            : nil

        let loadedModel = LoadedModel(
            modelSpec: modelSpec,
            rootURL: resolved,
            resources: resources,
            configs: configs,
            tokenizer: tokenizer,
            quantConfig: quantConfig,
            encoder: nil,
            transformer: nil,
            vae: nil
        )

        loaded = loadedModel
        return loadedModel
    }

    func ensureEncoderLoaded(
        model: inout LoadedModel,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws {
        if model.encoder != nil {
            return
        }

        let encoder = Qwen25VLEncoder.fromConfig(textEncoderConfig: model.configs.textEncoder)
        if let quantConfig = model.quantConfig {
            progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 0, totalSteps: 1))
            do {
                let encoderArrays = try QuantizedWeightLoader.loadArrays(from: model.resources.textEncoderWeightsURL)
                try PreQuantizedModelLoader.applyQuantizedLeafModules(
                    arrays: encoderArrays,
                    quantConfig: quantConfig,
                    to: encoder
                )
                try encoder.update(
                    parameters: ModuleParameters.unflattened(encoderArrays),
                    verify: [.shapeMismatch, .noUnusedKeys]
                )
            } catch {
                throw GeneratorError.modelLoadFailed(
                    "Pre-quantized encoder weights could not be applied. Rebuild the quantized image-edit weights with the companion quantization tool or use the full-precision model. Underlying error: \(error)"
                )
            }

            MLX.eval(encoder)
            Memory.clearCache()
            model.encoder = encoder
            return
        }

        progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 0, totalSteps: 2))
        try loadEncoderWeights(resources: model.resources, into: encoder)
        progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 1, totalSteps: 2))
        MLXNN.quantize(model: encoder, groupSize: 64, bits: 4) { _, module in
            if let linear = module as? Linear {
                let (_, inputDim) = linear.shape
                return inputDim % 64 == 0
            }
            return true
        }
        MLX.eval(encoder)
        Memory.clearCache()
        model.encoder = encoder
    }

    func ensureTransformerLoaded(
        model: inout LoadedModel,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws {
        if model.transformer != nil {
            return
        }

        if let quantConfig = model.quantConfig {
            do {
                progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 0, totalSteps: 1))
                let transformerWeightsURL = model.resources.rootURL
                    .appendingPathComponent("transformer")
                    .appendingPathComponent("model.safetensors")

                let transformerArrays = try QuantizedWeightLoader.loadArrays(from: transformerWeightsURL)
                let transformerFactory = DenseLayerFactory(arrays: transformerArrays, quantConfig: quantConfig)
                let transformer = MMDiT(config: model.configs.transformer, factory: transformerFactory)
                try transformer.update(
                    parameters: ModuleParameters.unflattened(transformerArrays),
                    verify: [.shapeMismatch, .noUnusedKeys]
                )

                MLX.eval(transformer)
                Memory.clearCache()
                model.transformer = transformer
                return
            } catch {
                throw GeneratorError.modelLoadFailed(
                    "Pre-quantized transformer weights could not be applied. Rebuild the quantized image-edit weights with the companion quantization tool or use the full-precision model. Underlying error: \(error)"
                )
            }
        }

        progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 0, totalSteps: 1))
        let transformer = MMDiT(config: model.configs.transformer)
        try loadTransformerWeights(resources: model.resources, into: transformer)
        MLXNN.quantize(model: transformer, groupSize: 64, bits: 4) { _, module in
            if let linear = module as? Linear {
                let (_, inputDim) = linear.shape
                return inputDim % 64 == 0
            }
            return true
        }
        MLX.eval(transformer)
        Memory.clearCache()
        model.transformer = transformer
    }

    func ensureVAELoaded(
        model: inout LoadedModel,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws {
        if model.vae != nil {
            return
        }

        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
        let vae = QwenImageEditVAE(config: model.configs.vae)
        try loadVAEWeights(resources: model.resources, into: vae)
        MLX.eval(vae)
        Memory.clearCache()
        model.vae = vae
    }

    func resolveModelRoot(
        modelSpec: String,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> URL {
        try await QwenImageEditRepository.resolveModelRoot(
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

    func loadEncoderWeights(
        resources: QwenImageEditResources,
        into model: Qwen25VLEncoder
    ) throws {
        let indexURL = resources.textEncoderWeightsIndexURL
        if FileManager.default.fileExists(atPath: indexURL.path) {
            try HFSafetensorsWeightsLoader.applyShardedWeights(
                indexURL: indexURL,
                to: model.underlyingTextEncoder,
                dtype: .bfloat16,
                verify: [.shapeMismatch],
                mapper: Self.encoderWeightMapper
            )
        } else if FileManager.default.fileExists(atPath: resources.textEncoderWeightsURL.path) {
            try HFSafetensorsWeightsLoader.applyWeights(
                url: resources.textEncoderWeightsURL,
                to: model.underlyingTextEncoder,
                dtype: .bfloat16,
                verify: [.shapeMismatch],
                mapper: Self.encoderWeightMapper
            )
        } else {
            throw HFSafetensorsWeightsLoader.LoaderError.indexFileMissing(indexURL)
        }
    }

    func loadTransformerWeights(
        resources: QwenImageEditResources,
        into model: MMDiT
    ) throws {
        let mapper = Self.transformerWeightMapper(config: model.config)
        let indexURL = resources.transformerWeightsIndexURL
        if FileManager.default.fileExists(atPath: indexURL.path) {
            try HFSafetensorsWeightsLoader.applyShardedWeights(
                indexURL: indexURL,
                to: model,
                dtype: .bfloat16,
                verify: [.shapeMismatch],
                mapper: mapper
            )
        } else if FileManager.default.fileExists(atPath: resources.transformerWeightsURL.path) {
            try HFSafetensorsWeightsLoader.applyWeights(
                url: resources.transformerWeightsURL,
                to: model,
                dtype: .bfloat16,
                verify: [.shapeMismatch],
                mapper: mapper
            )
        } else {
            throw HFSafetensorsWeightsLoader.LoaderError.indexFileMissing(indexURL)
        }
    }

    func loadVAEWeights(
        resources: QwenImageEditResources,
        into model: QwenImageEditVAE
    ) throws {
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.vaeWeightsURL,
            to: model.underlyingVAE,
            dtype: .bfloat16,
            verify: [.shapeMismatch],
            mapper: QwenImageEditVAE.weightMapper
        )
    }

    public static func loadTransformerWeights(
        into model: MMDiT,
        from weightFiles: [URL],
        config: QwenImageEditTransformerConfig
    ) async throws {
        let mapper = transformerWeightMapper(config: config)
        for url in weightFiles {
            try HFSafetensorsWeightsLoader.applyWeights(
                url: url,
                to: model,
                dtype: .bfloat16,
                verify: [.shapeMismatch],
                mapper: mapper
            )
        }
    }

    public static func loadEncoderWeights(
        into encoder: Qwen25VLEncoder,
        from weightFiles: [URL],
        config: QwenImageEditTextEncoderConfig
    ) async throws {
        let mapper = textEncoderWeightMapper(config: config)
        for url in weightFiles {
            try HFSafetensorsWeightsLoader.applyWeights(
                url: url,
                to: encoder,
                dtype: .bfloat16,
                verify: [.shapeMismatch],
                mapper: mapper
            )
        }
    }

    static func textEncoderWeightMapper(
        config: QwenImageEditTextEncoderConfig
    ) -> (String, MLXArray) -> [(String, MLXArray)] {
        return { rawKey, value in
            var key = rawKey
            if key.hasPrefix("model.") {
                key = "textEncoder.encoder." + String(key.dropFirst("model.".count))
            }
            if key.hasPrefix("visual.") {
                key = "visionTower." + String(key.dropFirst("visual.".count))
                if key.contains(".patch_embed.proj.") {
                    key = key.replacingOccurrences(of: ".patch_embed.proj.", with: ".patch_embed.")
                }
                if key.contains(".merger.") {
                    key = key.replacingOccurrences(of: ".merger.", with: ".patch_merger.")
                }
            }
            if key.hasPrefix("lm_head.") {
                return []
            }
            return [(key, value)]
        }
    }

    static func encoderWeightMapper(key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key.hasPrefix("model.") {
            let remainder = String(key.dropFirst("model.".count))
            return [("encoder.\(remainder)", value)]
        }
        return [(key, value)]
    }

    static func transformerWeightMapper(
        config: QwenImageEditTransformerConfig
    ) -> (String, MLXArray) -> [(String, MLXArray)] {
        return { rawKey, value in
            var key = rawKey

            if key.hasPrefix("img_in.") {
                return [("x_embedder." + String(key.dropFirst("img_in.".count)), value)]
            }
            if key.hasPrefix("txt_in.") {
                return [("context_embedder." + String(key.dropFirst("txt_in.".count)), value)]
            }
            if key.hasPrefix("txt_norm.") {
                return [(key, value)]
            }
            if key.hasPrefix("time_text_embed.timestep_embedder.linear_1.") {
                let suffix = String(key.dropFirst("time_text_embed.timestep_embedder.linear_1.".count))
                return [("t_embedder.mlp.0." + suffix, value)]
            }
            if key.hasPrefix("time_text_embed.timestep_embedder.linear_2.") {
                let suffix = String(key.dropFirst("time_text_embed.timestep_embedder.linear_2.".count))
                return [("t_embedder.mlp.1." + suffix, value)]
            }
            if key.contains(".attn.to_out.0.") {
                key = key.replacingOccurrences(of: ".attn.to_out.0.", with: ".attn.to_out.")
            }
            if key.contains(".img_mlp.net.0.proj.") {
                key = key.replacingOccurrences(of: ".img_mlp.net.0.proj.", with: ".ff.linear1.")
            }
            if key.contains(".img_mlp.net.2.") {
                key = key.replacingOccurrences(of: ".img_mlp.net.2.", with: ".ff.linear2.")
            }
            if key.contains(".txt_mlp.net.0.proj.") {
                key = key.replacingOccurrences(of: ".txt_mlp.net.0.proj.", with: ".ff_context.linear1.")
            }
            if key.contains(".txt_mlp.net.2.") {
                key = key.replacingOccurrences(of: ".txt_mlp.net.2.", with: ".ff_context.linear2.")
            }
            if key.contains(".img_mod.1.") {
                key = key.replacingOccurrences(of: ".img_mod.1.", with: ".adaLN_modulation.linear.")
            }
            if key.contains(".txt_mod.1.") {
                key = key.replacingOccurrences(of: ".txt_mod.1.", with: ".adaLN_modulation_context.linear.")
            }
            return [(key, value)]
        }
    }
}
