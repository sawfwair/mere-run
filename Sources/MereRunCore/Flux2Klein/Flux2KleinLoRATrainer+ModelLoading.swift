import Foundation
import MLX
import MLXNN
import MLXRandom

extension Flux2KleinLoRATrainer {
    // MARK: - Model Loading (copied from Flux2KleinGenerator)

    static func loadTransformerConfig(from transformerDir: URL) throws -> Flux2TransformerConfiguration {
        let configURL = transformerDir.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(Flux2TransformerConfig.self, from: data)

        return Flux2TransformerConfiguration(
            hiddenSize: config.numAttentionHeads * config.attentionHeadDim,
            numHeads: config.numAttentionHeads,
            headDim: config.attentionHeadDim,
            numLayers: config.numLayers,
            numSingleLayers: config.numSingleLayers,
            inChannels: config.inChannels,
            contextDim: config.jointAttentionDim,
            mlpRatio: config.mlpRatio,
            eps: config.eps,
            ropeTheta: config.ropeTheta,
            axesDimsRope: config.axesDimsRope
        )
    }

    static func loadTextEncoderConfig(from textEncoderDir: URL) throws -> QwenTextEncoderConfiguration {
        let configURL = textEncoderDir.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(Flux2TextEncoderConfig.self, from: data)

        return QwenTextEncoderConfiguration(
            vocabSize: config.vocabSize,
            hiddenSize: config.hiddenSize,
            numHiddenLayers: config.numHiddenLayers,
            numAttentionHeads: config.numAttentionHeads,
            numKeyValueHeads: config.numKeyValueHeads,
            intermediateSize: config.intermediateSize,
            ropeTheta: config.ropeTheta,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            rmsNormEps: config.rmsNormEps,
            headDim: config.headDim
        )
    }

    static func loadVAEConfig(from vaeDir: URL) throws -> VAEConfig {
        let configURL = vaeDir.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(Flux2VAEConfig.self, from: data)

        return VAEConfig(
            inChannels: config.inChannels,
            outChannels: config.outChannels,
            latentChannels: config.latentChannels,
            scalingFactor: config.scalingFactor ?? 1.0,
            shiftFactor: config.shiftFactor ?? 0.0,
            blockOutChannels: config.blockOutChannels,
            layersPerBlock: config.layersPerBlock,
            normNumGroups: config.normNumGroups,
            sampleSize: config.sampleSize ?? 1024,
            midBlockAddAttention: config.midBlockAddAttention,
            useQuantConv: config.useQuantConv ?? false,
            usePostQuantConv: config.usePostQuantConv ?? false
        )
    }

    static func loadBatchNormStats(from url: URL) throws -> (mean: MLXArray, variance: MLXArray) {
        let weights = try MLX.loadArrays(url: url)
        guard let bnMean = weights["bn.running_mean"],
              let bnVar = weights["bn.running_var"] else {
            throw Flux2KleinLoRATrainerError.missingBatchNormStats
        }
        return (bnMean, bnVar)
    }

    static func loadTokenizer(from url: URL) throws -> QwenTokenizer {
        try QwenTokenizer.load(from: url, maxLengthOverride: 512)
    }

    static func loadTransformerWeights(
        from url: URL,
        to transformer: Flux2Transformer2DModel,
        quantization: ModelWeightsLoader.QuantizationParams?
    ) throws {
        let singleFileURL = url.appendingPathComponent("diffusion_pytorch_model.safetensors")
        let indexURL = url.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: singleFileURL.path) {
            try ModelWeightsLoader.applyHFSafetensors(
                indexURL: indexURL,
                singleURL: singleFileURL,
                to: transformer,
                dtype: .bfloat16,
                verify: .noUnusedKeys,
                quantization: quantization
            )
            return
        }

        let files = try ModelWeightsLoader.safetensorsShards(in: url)
        guard !files.isEmpty else {
            throw NSError(domain: "Flux2KleinLoRATrainer", code: 1, userInfo: [NSLocalizedDescriptionKey: "No transformer weights found at \(url.path)"])
        }

        let mfluxKeyMapper: (String) -> String = { key in
            if key.hasPrefix("time_guidance_embed.linear_") {
                return key.replacingOccurrences(
                    of: "time_guidance_embed.linear_",
                    with: "time_guidance_embed.timestep_embedder.linear_"
                )
            }
            if key.hasPrefix("transformer_blocks.") && key.contains(".attn.to_out.") && !key.contains(".to_out.0.") {
                return key.replacingOccurrences(of: ".attn.to_out.", with: ".attn.to_out.0.")
            }
            return key
        }

        try ModelWeightsLoader.applySafetensorsShards(
            files: files,
            to: transformer,
            dtype: .bfloat16,
            verify: .noUnusedKeys,
            mapper: { key, value in
                [(mfluxKeyMapper(key), value)]
            },
            keyMapper: mfluxKeyMapper,
            quantization: quantization
        )
    }

    static func loadTextEncoderWeights(
        from url: URL,
        to encoder: QwenTextEncoder,
        quantization: ModelWeightsLoader.QuantizationParams?
    ) async throws {
        let indexURL = url.appendingPathComponent("model.safetensors.index.json")
        let singleFileURL = url.appendingPathComponent("model.safetensors")

        let mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            // The text encoder is used for hidden states only; some Qwen3 checkpoints
            // include a language-model output head that is not part of QwenTextEncoder.
            if key == "lm_head" || key.hasPrefix("lm_head.") || key.hasPrefix("model.lm_head") {
                return []
            }
            var mappedKey = key
            if key.hasPrefix("model.") {
                mappedKey = key.replacingOccurrences(of: "model.", with: "encoder.")
            } else if !key.hasPrefix("encoder.") {
                mappedKey = "encoder." + key
            }
            return [(mappedKey, value)]
        }

        let keyMapper: (String) -> String = { key in
            if key == "lm_head" || key.hasPrefix("lm_head.") || key.hasPrefix("model.lm_head") {
                return "__unused__." + key
            }
            if key.hasPrefix("model.") {
                return "encoder." + String(key.dropFirst("model.".count))
            }
            if key.hasPrefix("encoder.") {
                return key
            }
            return "encoder." + key
        }

        let fm = FileManager.default
        if fm.fileExists(atPath: indexURL.path) || fm.fileExists(atPath: singleFileURL.path) {
            try ModelWeightsLoader.applyHFSafetensors(
                indexURL: indexURL,
                singleURL: singleFileURL,
                to: encoder,
                dtype: .bfloat16,
                verify: .noUnusedKeys,
                mapper: mapper,
                keyMapper: keyMapper,
                quantization: quantization
            )
            return
        }

        let files = try ModelWeightsLoader.safetensorsShards(in: url)
        guard !files.isEmpty else {
            throw NSError(domain: "Flux2KleinLoRATrainer", code: 1, userInfo: [NSLocalizedDescriptionKey: "No text encoder weights found at \(url.path)"])
        }

        try ModelWeightsLoader.applySafetensorsShards(
            files: files,
            to: encoder,
            dtype: .bfloat16,
            verify: .noUnusedKeys,
            mapper: mapper,
            keyMapper: keyMapper,
            quantization: quantization
        )
    }

    static func loadVAEWeights(from url: URL, to vae: AutoencoderKL) throws {
        try HFSafetensorsWeightsLoader.applyWeights(
            url: url,
            to: vae,
            verify: [.noUnusedKeys, .shapeMismatch],
            mapper: { key, value in
                if key.hasPrefix("bn.") {
                    return []
                }
                if value.ndim == 4 && key.contains("conv") {
                    return [(key, HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(value))]
                }
                return [(key, value)]
            }
        )
    }
}
