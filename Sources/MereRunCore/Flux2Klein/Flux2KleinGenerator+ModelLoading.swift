import Foundation
import MLX
import MLXRandom
import MLXNN

extension Flux2KleinGenerator {

    // MARK: - Chat Model Loading

    /// Load only text encoder and tokenizer (for chat without image generation)
    func loadTextEncoderOnly(
        from path: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let debugLog = MereRunRuntimeDebug.logger(keys: ["MERERUN_FLUX2_DEBUG"], prefix: "[Flux2KleinGenerator]")
        let modelURL = URL(fileURLWithPath: path).standardizedFileURL
        let manifest = try MereRunModelManifest.loadRequired(from: modelURL)
        let componentResolver = ModelComponentResolver(modelRootURL: modelURL, manifest: manifest)

        let textEncoderComponent = try componentResolver.resolveDirectory(for: .textEncoder, fallbackLocalPath: "text_encoder")
        let tokenizerComponent = try componentResolver.resolveDirectory(for: .tokenizer, fallbackLocalPath: "tokenizer")
        let quantization = try ModelWeightsLoader.QuantizationParams.fromManifest(textEncoderComponent.sourceManifest)

        // Load text encoder
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading text encoder"))
        let textEncoderConfig = try loadTextEncoderConfig(from: textEncoderComponent.directoryURL)
        textEncoder = QwenTextEncoder(configuration: textEncoderConfig)

        try await loadTextEncoderWeights(from: textEncoderComponent.directoryURL, to: textEncoder!, quantization: quantization)

        // Load tokenizer
        tokenizer = try loadTokenizer(from: tokenizerComponent.directoryURL)

        loadedModelPath = path
        loadedManifest = manifest
        loadedQuantization = quantization
        currentTextLoRA = nil
        debugLog?("chat: text encoder loaded from \(path)")
    }

    /// Load a standalone Qwen3-Instruct model for chat (config, weights, tokenizer all at root).
    func loadStandaloneChatModel(
        from path: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let debugLog = MereRunRuntimeDebug.logger(keys: ["MERERUN_FLUX2_DEBUG"], prefix: "[Flux2KleinGenerator]")
        let modelURL = URL(fileURLWithPath: path).standardizedFileURL

        // Config and weights live at root (no text_encoder/ subdir)
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading chat model"))

        // Read raw HF config to extract quantization_config
        let configURL = modelURL.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let rawConfig = try JSONDecoder().decode(Flux2TextEncoderConfig.self, from: configData)

        let quantization: ModelWeightsLoader.QuantizationParams?
        if let qc = rawConfig.quantizationConfig,
           let bits = qc.bits, let groupSize = qc.groupSize {
            quantization = ModelWeightsLoader.QuantizationParams(bits: bits, groupSize: groupSize)
        } else {
            quantization = nil
        }

        let config = QwenTextEncoderConfiguration(
            vocabSize: rawConfig.vocabSize,
            hiddenSize: rawConfig.hiddenSize,
            numHiddenLayers: rawConfig.numHiddenLayers,
            numAttentionHeads: rawConfig.numAttentionHeads,
            numKeyValueHeads: rawConfig.numKeyValueHeads,
            intermediateSize: rawConfig.intermediateSize,
            ropeTheta: rawConfig.ropeTheta,
            maxPositionEmbeddings: rawConfig.maxPositionEmbeddings,
            rmsNormEps: rawConfig.rmsNormEps,
            headDim: rawConfig.headDim
        )

        textEncoder = QwenTextEncoder(configuration: config)
        try await loadTextEncoderWeights(from: modelURL, to: textEncoder!, quantization: quantization)

        // Tokenizer also at root (no tokenizer/ subdir)
        tokenizer = try loadTokenizer(from: modelURL)

        loadedModelPath = path
        loadedManifest = nil
        loadedQuantization = quantization
        currentTextLoRA = nil
        debugLog?("chat: standalone model loaded from \(path)")
    }

    /// Apply text LoRA (game cartridge) to text encoder for chat
    func applyTextLoRAIfNeeded(
        _ lora: LoRA?,
        modelPath: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let debugLog = MereRunRuntimeDebug.logger(keys: ["MERERUN_FLUX2_DEBUG"], prefix: "[Flux2KleinGenerator]")
        guard lora != currentTextLoRA else { return }

        // If switching LoRAs, need to reload base weights first
        if currentTextLoRA != nil {
            debugLog?("chat: reloading base text encoder")
            let modelURL = URL(fileURLWithPath: modelPath).standardizedFileURL

            if let manifest = try? MereRunModelManifest.loadRequired(from: modelURL) {
                // Klein model path: weights in text_encoder/ subdir
                let componentResolver = ModelComponentResolver(modelRootURL: modelURL, manifest: manifest)
                let textEncoderComponent = try componentResolver.resolveDirectory(for: .textEncoder, fallbackLocalPath: "text_encoder")
                let quantization = try ModelWeightsLoader.QuantizationParams.fromManifest(textEncoderComponent.sourceManifest)
                try await loadTextEncoderWeights(from: textEncoderComponent.directoryURL, to: textEncoder!, quantization: quantization)
                loadedManifest = manifest
                loadedQuantization = quantization
            } else {
                // Standalone model path: weights at root
                try await loadTextEncoderWeights(from: modelURL, to: textEncoder!, quantization: loadedQuantization)
            }
        }

        guard let lora else {
            currentTextLoRA = nil
            return
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading cartridge"))
        debugLog?("chat: loading text LoRA")

        let weights = try await QwenTextLoRAWeightLoader.load(from: lora)
        let applied = QwenTextLoRAApplicator.mergeIntoTextEncoder(
            textEncoder!,
            loraWeights: weights,
            scale: 1.0
        )

        if applied == 0 {
            throw LoRAError.invalidFormat("No matching Qwen text encoder layers found for this cartridge.")
        }

        currentTextLoRA = lora
        debugLog?("chat: applied text LoRA with \(applied) layers")
    }


    // MARK: - Model Loading

    func loadModels(from path: String, progressHandler: (@Sendable (GenerationProgress) -> Void)?) async throws {
        let modelURL = URL(fileURLWithPath: path).standardizedFileURL
        let manifest = try MereRunModelManifest.loadRequired(from: modelURL)

        let componentResolver = ModelComponentResolver(modelRootURL: modelURL, manifest: manifest)
        let transformerComponent = try componentResolver.resolveDirectory(for: .transformer, fallbackLocalPath: "transformer")
        let textEncoderComponent = try componentResolver.resolveDirectory(for: .textEncoder, fallbackLocalPath: "text_encoder")
        let tokenizerComponent = try componentResolver.resolveDirectory(for: .tokenizer, fallbackLocalPath: "tokenizer")
        let vaeComponent = try componentResolver.resolveDirectory(for: .vae, fallbackLocalPath: "vae")

        let transformerQuantization = try ModelWeightsLoader.QuantizationParams.fromManifest(transformerComponent.sourceManifest)
        let textEncoderQuantization = try ModelWeightsLoader.QuantizationParams.fromManifest(textEncoderComponent.sourceManifest)

        // Load transformer
        // Shared shard discovery resolves symlinked component directories before listing them.
        progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 0, totalSteps: 1))
        let transformerConfig = try loadTransformerConfig(from: transformerComponent.directoryURL)
        transformer = Flux2Transformer2DModel(config: transformerConfig)

        try loadTransformerWeights(from: transformerComponent.directoryURL, to: transformer!, quantization: transformerQuantization)
        progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 1, totalSteps: 1))

        // Load text encoder
        progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 0, totalSteps: 1))
        let textEncoderConfig = try loadTextEncoderConfig(from: textEncoderComponent.directoryURL)
        textEncoder = QwenTextEncoder(configuration: textEncoderConfig)

        try await loadTextEncoderWeights(from: textEncoderComponent.directoryURL, to: textEncoder!, quantization: textEncoderQuantization)
        progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 1, totalSteps: 1))

        // Load tokenizer
        tokenizer = try loadTokenizer(from: tokenizerComponent.directoryURL)

        // Load VAE
        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
        let vaeConfig = try loadVAEConfig(from: vaeComponent.directoryURL)
        vae = AutoencoderKL(configuration: vaeConfig)

        let vaeWeightsURL = vaeComponent.directoryURL.appendingPathComponent("diffusion_pytorch_model.safetensors").resolvingSymlinksInPath()

        // First, load BN running stats directly from safetensors
        let bnWeights = try loadBatchNormStats(from: vaeWeightsURL)
        bnRunningMean = bnWeights.mean
        bnRunningVar = bnWeights.variance

        try HFSafetensorsWeightsLoader.applyWeights(
            url: vaeWeightsURL,
            to: vae!,
            verify: [.noUnusedKeys, .shapeMismatch],
            mapper: { key, value in
                // Drop PyTorch BatchNorm buffers (we loaded them separately)
                if key.hasPrefix("bn.") {
                    return []
                }
                // Transpose conv weights from PyTorch [OIHW] to MLX [OHWI] format
                if value.ndim == 4 && key.contains("conv") {
                    return [(key, HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(value))]
                }
                return [(key, value)]
            }
        )
        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 1, totalSteps: 1))

        loadedModelPath = path
        loadedManifest = manifest
        loadedQuantization = transformerQuantization
        currentLoRA = nil
        transformerLoRALayers = nil
        transformerLoRARankSignature = nil
        compiledTransformer = nil
        compiledTransformerNeedsWarmup = true
    }

    func applyLoRAIfNeeded(
        _ lora: LoRA?,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws {
        guard lora != currentLoRA else { return }
        guard transformer != nil else { return }

        // Clearing LoRA (keep injected wrappers but disable them).
        guard let lora else {
            if let layers = transformerLoRALayers {
                for layer in layers.values {
                    layer.isActive = false
                }
            }
            currentLoRA = nil
            progressHandler?(GenerationProgress(stage: .loadingLoRA, stepIndex: 1, totalSteps: 1))
            return
        }

        progressHandler?(GenerationProgress(stage: .loadingLoRA, stepIndex: 0, totalSteps: 1))
        let loraWeights = try await LoRAWeightLoader.load(from: lora)
        let targetRank = loraWeights.rank
        let targetRanks = loraWeights.targetRanks.isEmpty ? nil : loraWeights.targetRanks
        let targetRankSignature = Self.loraRankSignature(defaultRank: targetRank, targetRanks: targetRanks)

        if transformerLoRALayers == nil {
            transformerLoRALayers = try Flux2LoRAInjector.inject(
                into: transformer!,
                rank: targetRank,
                alpha: loraWeights.alpha,
                targetRanks: targetRanks,
                zeroInitUp: true
            )
            transformerLoRARankSignature = targetRankSignature
            if let layers = transformerLoRALayers {
                for layer in layers.values {
                    layer.isActive = false
                }
            }

            // Module structure changed; invalidate compilation.
            compiledTransformer = nil
            compiledTransformerNeedsWarmup = true
        } else if transformerLoRARankSignature != targetRankSignature {
            transformerLoRALayers = try Flux2LoRAInjector.inject(
                into: transformer!,
                rank: targetRank,
                alpha: loraWeights.alpha,
                targetRanks: targetRanks,
                zeroInitUp: true,
                allowReinjection: true
            )
            transformerLoRARankSignature = targetRankSignature
            if let layers = transformerLoRALayers {
                for layer in layers.values {
                    layer.isActive = false
                }
            }

            // Module structure changed; invalidate compilation.
            compiledTransformer = nil
            compiledTransformerNeedsWarmup = true
        }

        guard let layers = transformerLoRALayers else {
            throw LoRAError.invalidFormat("Failed to inject Flux2Klein LoRA layers.")
        }

        var applied = 0
        for (path, layer) in layers {
            guard let weights = loraWeights.weights[path] else {
                layer.isActive = false
                continue
            }

            layer.loraDown = weights.down.asType(.float32)
            layer.loraUp = weights.up.asType(.float32)
            layer.isActive = true
            applied += 1
        }

        // Apply user scale by scaling lora_down (delta scales linearly).
        let userScale = Self.loraScale(for: lora)
        if userScale != 1 {
            let s = MLXArray(userScale)
            for layer in layers.values where layer.isActive {
                layer.loraDown = layer.loraDown * s
            }
        }
        progressHandler?(GenerationProgress(stage: .loadingLoRA, stepIndex: 1, totalSteps: 1))

        guard applied > 0 else {
            throw LoRAError.invalidFormat(Self.noMatchingTransformerLayersMessage)
        }

        currentLoRA = lora
    }

    private static func loraScale(for lora: LoRA) -> Float {
        switch lora {
        case .local(_, let scale):
            return Float(scale)
        case .remote(_, let scale):
            return Float(scale)
        }
    }

    private static func loraRankSignature(defaultRank: Int, targetRanks: [String: Int]?) -> String {
        guard let targetRanks, !targetRanks.isEmpty else {
            return "default:\(defaultRank)"
        }
        return targetRanks
            .sorted { lhs, rhs in lhs.key < rhs.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: ";")
    }

    private static let noMatchingTransformerLayersMessage = "No matching transformer layers found for this LoRA."

    static let compileEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_FLUX2_COMPILE"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }()

    func transformerForward(
        _ transformer: Flux2Transformer2DModel,
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        timestep: MLXArray,
        imgIds: MLXArray,
        txtIds: MLXArray
    ) -> MLXArray {
        guard Self.compileEnabled else {
            return transformer(
                hiddenStates: hiddenStates,
                encoderHiddenStates: encoderHiddenStates,
                timestep: timestep,
                imgIds: imgIds,
                txtIds: txtIds,
                guidance: nil
            )
        }

        if compiledTransformer == nil {
            compiledTransformer = compile(inputs: [transformer], outputs: [transformer]) { inputs in
                let output = transformer(
                    hiddenStates: inputs[0],
                    encoderHiddenStates: inputs[1],
                    timestep: inputs[2],
                    imgIds: inputs[3],
                    txtIds: inputs[4],
                    guidance: nil
                )
                return [output]
            }
            compiledTransformerNeedsWarmup = true
        }

        guard let compiledTransformer else {
            return transformer(
                hiddenStates: hiddenStates,
                encoderHiddenStates: encoderHiddenStates,
                timestep: timestep,
                imgIds: imgIds,
                txtIds: txtIds,
                guidance: nil
            )
        }

        if compiledTransformerNeedsWarmup {
            _ = compiledTransformer([hiddenStates, encoderHiddenStates, timestep, imgIds, txtIds])
            compiledTransformerNeedsWarmup = false
        }
        return compiledTransformer([hiddenStates, encoderHiddenStates, timestep, imgIds, txtIds])[0]
    }

    private func loadTransformerConfig(from transformerDir: URL) throws -> Flux2TransformerConfiguration {
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

    private func loadTextEncoderConfig(from textEncoderDir: URL) throws -> QwenTextEncoderConfiguration {
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

    private func loadVAEConfig(from vaeDir: URL) throws -> VAEConfig {
        let configURL = vaeDir.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(Flux2VAEConfig.self, from: data)

        return VAEConfig(
            inChannels: config.inChannels,
            outChannels: config.outChannels,
            latentChannels: config.latentChannels,
            // FLUX.2 Klein VAE uses scaling=1.0, shift=0.0 (not the SD3 defaults)
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

    private func loadTransformerWeights(
        from url: URL,
        to transformer: Flux2Transformer2DModel,
        quantization: ModelWeightsLoader.QuantizationParams?
    ) throws {
        // Key mapper for mflux format -> our format.
        // mflux uses to_out.X but we use to_out.0.X (array) for transformer_blocks only.
        // single_transformer_blocks uses to_out as single Linear (no array).
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

        // Check for single file first (our format)
        let singleFileURL = url.appendingPathComponent("diffusion_pytorch_model.safetensors")
        let indexURL = url.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: singleFileURL.path) {
            try ModelWeightsLoader.applyHFSafetensors(
                indexURL: indexURL,
                singleURL: singleFileURL,
                to: transformer,
                dtype: .bfloat16,
                verify: .noUnusedKeys,
                mapper: { key, value in [(mfluxKeyMapper(key), value)] },
                keyMapper: mfluxKeyMapper,
                quantization: quantization
            )
            return
        }

        let files = try ModelWeightsLoader.safetensorsShards(in: url)
        guard !files.isEmpty else {
            throw NSError(domain: "Flux2KleinGenerator", code: 1, userInfo: [NSLocalizedDescriptionKey: "No transformer weights found at \(url.path)"])
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

    private func loadTextEncoderWeights(
        from url: URL,
        to encoder: QwenTextEncoder,
        quantization: ModelWeightsLoader.QuantizationParams?
    ) async throws {
        let indexURL = url.appendingPathComponent("model.safetensors.index.json")
        let singleFileURL = url.appendingPathComponent("model.safetensors")

        let mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            // The Qwen3 text encoder used as an embedder only needs hidden states; the
            // language-model output head (lm_head.*) is not part of the encoder and ships
            // in some checkpoints (e.g. FLUX.2-klein-9B's 8B Qwen3 embedder). Drop it so
            // Module.update(verify: .noUnusedKeys) doesn't reject the load.
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
            throw NSError(domain: "Flux2KleinGenerator", code: 1, userInfo: [NSLocalizedDescriptionKey: "No text encoder weights found at \(url.path)"])
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

    private func loadTokenizer(from url: URL) throws -> QwenTokenizer {
        // 2048 for chat context (system prompt + game state can be large)
        // 512 was too small and caused truncation
        try QwenTokenizer.load(from: url, maxLengthOverride: 2048)
    }

    /// Load BatchNorm running statistics from VAE weights
    private func loadBatchNormStats(from url: URL) throws -> (mean: MLXArray, variance: MLXArray) {
        let weights = try MLX.loadArrays(url: url)

        guard let bnMean = weights["bn.running_mean"],
              let bnVar = weights["bn.running_var"] else {
            throw Flux2Error.missingBatchNormStats
        }

        return (mean: bnMean, variance: bnVar)
    }


}
