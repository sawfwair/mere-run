import Foundation
import MLX
import MLXRandom
import MLXNN

extension Flux2KleinGeneratoriOS {

    // MARK: - Config Loading

    func loadTransformerConfig(from transformerDirURL: URL) throws -> Flux2TransformerConfiguration {
        let configURL = transformerDirURL.appendingPathComponent("config.json")
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

    func loadTextEncoderConfig(from textEncoderDirURL: URL) throws -> QwenTextEncoderConfiguration {
        let configURL = textEncoderDirURL.appendingPathComponent("config.json")
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

    func loadVAEConfig(from vaeDirURL: URL) throws -> VAEConfig {
        let configURL = vaeDirURL.appendingPathComponent("config.json")
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

    // MARK: - Weight Loading (iOS Streaming)

    /// Build transformer from weights file - factory method for iOS
    /// For quantized weights, builds with QuantizedLinear from scratch
    func loadTransformer(
        config: Flux2TransformerConfiguration,
        from url: URL,
        quantization: ModelWeightsLoader.QuantizationParams?
    ) throws -> Flux2Transformer2DModel {
        let singleFileURL = url.appendingPathComponent("diffusion_pytorch_model.safetensors")
        if FileManager.default.fileExists(atPath: singleFileURL.path) {
            return try loadTransformerFromFile(config: config, url: singleFileURL, quantization: quantization)
        }

        // Sharded weights not supported for iOS quantized loading
        throw NSError(domain: "Flux2KleinGeneratoriOS", code: 1, userInfo: [NSLocalizedDescriptionKey: "iOS requires single-file transformer weights"])
    }

    /// Load transformer from a single safetensors file
    private func loadTransformerFromFile(
        config: Flux2TransformerConfiguration,
        url: URL,
        quantization: ModelWeightsLoader.QuantizationParams?
    ) throws -> Flux2Transformer2DModel {
        // Memory-map the file to avoid loading entire file into RAM
        let fileData = try Data(contentsOf: url, options: .mappedIfSafe)

        let tensorMetadata = try SafetensorsStreamingLoader.metadata(fileData: fileData, fileURL: url)

        // Check if quantized by looking for .scales keys
        let isQuantized = tensorMetadata.keys.contains { $0.hasSuffix(".scales") }

        if isQuantized {
            guard let quantization else {
                throw ModelWeightsLoader.LoaderError.invalidQuantizationMetadata(
                    "Quantized transformer weights detected, but quantization params were not provided (missing \(MereRunModelManifest.filename))."
                )
            }
            return try buildQuantizedTransformer(
                config: config,
                tensorMetadata: tensorMetadata,
                fileData: fileData,
                quantization: quantization
            )
        } else {
            // Non-quantized: create transformer and load weights
            let transformer = Flux2Transformer2DModel(config: config)
            try applyWeightsStreaming(tensorMetadata: tensorMetadata, fileData: fileData, to: transformer)
            return transformer
        }
    }

    /// Apply non-quantized weights in streaming fashion
    private func applyWeightsStreaming(
        tensorMetadata: [String: SafetensorsStreamingLoader.TensorMetadata],
        fileData: Data,
        to model: Module
    ) throws {
        var updates: [(String, MLXArray)] = []
        let batchSize = 50  // Apply weights in batches to balance memory vs overhead

        for (key, metadata) in tensorMetadata {
            let tensorData = fileData.subdata(in: metadata.startOffset..<metadata.endOffset)

            let array = MLXArray(tensorData, metadata.shape, dtype: metadata.dtype)
            let castArray = HFSafetensorsWeightsLoader.castIfNeeded(array, dtype: .bfloat16)
            updates.append((key, castArray))

            // Apply in batches
            if updates.count >= batchSize {
                try model.update(parameters: ModuleParameters.unflattened(updates), verify: .none)
                updates.removeAll(keepingCapacity: true)
                Memory.clearCache()
            }
        }

        // Apply remaining
        if !updates.isEmpty {
            try model.update(parameters: ModuleParameters.unflattened(updates), verify: .none)
        }
    }

    /// Build a quantized transformer from weights file
    /// Returns a new transformer with QuantizedLinear modules
    private func buildQuantizedTransformer(
        config: Flux2TransformerConfiguration,
        tensorMetadata: [String: SafetensorsStreamingLoader.TensorMetadata],
        fileData: Data,
        quantization: ModelWeightsLoader.QuantizationParams
    ) throws -> Flux2Transformer2DModel {
        // Helper to load a tensor from file
        func loadTensor(_ key: String) -> MLXArray? {
            let resolvedKey: String
            if tensorMetadata[key] != nil {
                resolvedKey = key
            } else {
                let mfluxKey = key.replacingOccurrences(of: ".attn.to_out.0.", with: ".attn.to_out.")
                guard tensorMetadata[mfluxKey] != nil else {
                    return nil
                }
                resolvedKey = mfluxKey
            }
            guard let metadata = tensorMetadata[resolvedKey] else {
                return nil
            }
            let tensorData = fileData.subdata(in: metadata.startOffset..<metadata.endOffset)
            return MLXArray(tensorData, metadata.shape, dtype: metadata.dtype)
        }

        // Helper to create QuantizedLinear from weights
        func makeQuantizedLinear(_ basePath: String) -> QuantizedLinear? {
            guard let weight = loadTensor("\(basePath).weight"),
                  let scales = loadTensor("\(basePath).scales") else {
                return nil
            }
            let biases = loadTensor("\(basePath).biases")
            return PortableQuantizedLinear(
                weight: weight,
                bias: nil,
                scales: scales,
                biases: biases,
                groupSize: quantization.groupSize,
                bits: quantization.bits
            )
        }

        // Load non-quantized parameter (for norms, etc)
        func loadParam(_ key: String) -> MLXArray? {
            guard let tensor = loadTensor(key) else { return nil }
            return HFSafetensorsWeightsLoader.castIfNeeded(tensor, dtype: .bfloat16)
        }

        // 1. Build all transformer_blocks with QuantizedLinear
        var jointBlocks: [Flux2TransformerBlock] = []
        for i in 0..<config.numLayers {
            let block = Flux2TransformerBlock(config: config, placeholder: true)
            updateJointBlock(
                block: block,
                index: i,
                loadTensor: loadTensor,
                loadParam: loadParam,
                makeQuantizedLinear: makeQuantizedLinear
            )
            jointBlocks.append(block)
            Memory.clearCache()
        }

        // 2. Build all single_transformer_blocks with QuantizedLinear
        var singleBlocks: [Flux2SingleTransformerBlock] = []
        for i in 0..<config.numSingleLayers {
            let block = Flux2SingleTransformerBlock(config: config, placeholder: true)
            updateSingleBlock(
                block: block,
                index: i,
                loadTensor: loadTensor,
                loadParam: loadParam,
                makeQuantizedLinear: makeQuantizedLinear
            )
            singleBlocks.append(block)
            Memory.clearCache()
        }

        // 3. Create transformer with pre-built blocks
        let transformer = Flux2Transformer2DModel(
            config: config,
            transformerBlocks: jointBlocks,
            singleTransformerBlocks: singleBlocks
        )

        // 4. Update root-level modules with QuantizedLinear
        var rootModuleUpdates: [(String, Module)] = []

        if let linear = makeQuantizedLinear("x_embedder") {
            rootModuleUpdates.append(("x_embedder", linear))
        }
        if let linear = makeQuantizedLinear("context_embedder") {
            rootModuleUpdates.append(("context_embedder", linear))
        }
        if let linear = makeQuantizedLinear("proj_out") {
            rootModuleUpdates.append(("proj_out", linear))
        }
        if let linear = makeQuantizedLinear("double_stream_modulation_img.linear") {
            rootModuleUpdates.append(("double_stream_modulation_img.linear", linear))
        }
        if let linear = makeQuantizedLinear("double_stream_modulation_txt.linear") {
            rootModuleUpdates.append(("double_stream_modulation_txt.linear", linear))
        }
        if let linear = makeQuantizedLinear("single_stream_modulation.linear") {
            rootModuleUpdates.append(("single_stream_modulation.linear", linear))
        }
        if let linear = makeQuantizedLinear("norm_out.linear") {
            rootModuleUpdates.append(("norm_out.linear", linear))
        }

        // Timestep embedder
        if let linear = makeQuantizedLinear("time_guidance_embed.timestep_embedder.linear_1") {
            rootModuleUpdates.append(("time_guidance_embed.timestep_embedder.linear_1", linear))
        }
        if let linear = makeQuantizedLinear("time_guidance_embed.timestep_embedder.linear_2") {
            rootModuleUpdates.append(("time_guidance_embed.timestep_embedder.linear_2", linear))
        }

        if !rootModuleUpdates.isEmpty {
            transformer.update(modules: ModuleChildren.unflattened(rootModuleUpdates))
            Memory.clearCache()
        }

        return transformer
    }

    /// Update an existing joint transformer block with quantized weights
    private func updateJointBlock(
        block: Flux2TransformerBlock,
        index: Int,
        loadTensor: (String) -> MLXArray?,
        loadParam: (String) -> MLXArray?,
        makeQuantizedLinear: (String) -> QuantizedLinear?
    ) {
        let prefix = "transformer_blocks.\(index)"

        // Build module updates
        var moduleUpdates: [(String, Module)] = []

        // Image stream attention
        if let q = makeQuantizedLinear("\(prefix).attn.to_q") {
            moduleUpdates.append(("attn.to_q", q))
        }
        if let k = makeQuantizedLinear("\(prefix).attn.to_k") {
            moduleUpdates.append(("attn.to_k", k))
        }
        if let v = makeQuantizedLinear("\(prefix).attn.to_v") {
            moduleUpdates.append(("attn.to_v", v))
        }
        // toOut is an array [Linear], update element 0
        if let out = makeQuantizedLinear("\(prefix).attn.to_out.0") {
            moduleUpdates.append(("attn.to_out.0", out))
        }

        // Context stream attention
        if let q = makeQuantizedLinear("\(prefix).attn.add_q_proj") {
            moduleUpdates.append(("attn.add_q_proj", q))
        }
        if let k = makeQuantizedLinear("\(prefix).attn.add_k_proj") {
            moduleUpdates.append(("attn.add_k_proj", k))
        }
        if let v = makeQuantizedLinear("\(prefix).attn.add_v_proj") {
            moduleUpdates.append(("attn.add_v_proj", v))
        }
        if let out = makeQuantizedLinear("\(prefix).attn.to_add_out") {
            moduleUpdates.append(("attn.to_add_out", out))
        }

        // FFN image
        if let lin = makeQuantizedLinear("\(prefix).ff.linear_in") {
            moduleUpdates.append(("ff.linear_in", lin))
        }
        if let lin = makeQuantizedLinear("\(prefix).ff.linear_out") {
            moduleUpdates.append(("ff.linear_out", lin))
        }

        // FFN context
        if let lin = makeQuantizedLinear("\(prefix).ff_context.linear_in") {
            moduleUpdates.append(("ff_context.linear_in", lin))
        }
        if let lin = makeQuantizedLinear("\(prefix).ff_context.linear_out") {
            moduleUpdates.append(("ff_context.linear_out", lin))
        }

        if !moduleUpdates.isEmpty {
            block.update(modules: ModuleChildren.unflattened(moduleUpdates))
        }

        // Load RMSNorm weights as parameters
        var normParams: [(String, MLXArray)] = []
        for normPath in ["attn.norm_q.weight", "attn.norm_k.weight",
                         "attn.norm_added_q.weight", "attn.norm_added_k.weight"] {
            if let w = loadParam("\(prefix).\(normPath)") {
                normParams.append((normPath, w))
            }
        }
        if !normParams.isEmpty {
            _ = try? block.update(parameters: ModuleParameters.unflattened(normParams), verify: .none)
        }
    }

    /// Update an existing single transformer block with quantized weights
    private func updateSingleBlock(
        block: Flux2SingleTransformerBlock,
        index: Int,
        loadTensor: (String) -> MLXArray?,
        loadParam: (String) -> MLXArray?,
        makeQuantizedLinear: (String) -> QuantizedLinear?
    ) {
        let prefix = "single_transformer_blocks.\(index)"

        // Build module updates
        var moduleUpdates: [(String, Module)] = []

        if let qkvMlp = makeQuantizedLinear("\(prefix).attn.to_qkv_mlp_proj") {
            moduleUpdates.append(("attn.to_qkv_mlp_proj", qkvMlp))
        }
        if let out = makeQuantizedLinear("\(prefix).attn.to_out") {
            moduleUpdates.append(("attn.to_out", out))
        }

        if !moduleUpdates.isEmpty {
            block.update(modules: ModuleChildren.unflattened(moduleUpdates))
        }

        // Load RMSNorm weights as parameters
        var normParams: [(String, MLXArray)] = []
        for normPath in ["attn.norm_q.weight", "attn.norm_k.weight"] {
            if let w = loadParam("\(prefix).\(normPath)") {
                normParams.append((normPath, w))
            }
        }
        if !normParams.isEmpty {
            _ = try? block.update(parameters: ModuleParameters.unflattened(normParams), verify: .none)
        }
    }

    func loadTextEncoderWeights(
        from url: URL,
        to encoder: QwenTextEncoder,
        quantization: ModelWeightsLoader.QuantizationParams?
    ) async throws {
        let mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
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

        let singleFileURL = url.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: singleFileURL.path) {
            do {
                let weights = try MLX.loadArrays(url: singleFileURL)
                if HFSafetensorsWeightsLoader.isQuantized(weights) {
                    guard let quantization else {
                        throw ModelWeightsLoader.LoaderError.invalidQuantizationMetadata(
                            "Quantized text encoder weights detected, but quantization params were not provided (missing \(MereRunModelManifest.filename))."
                        )
                    }
                    try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                        weights,
                        to: encoder,
                        groupSize: quantization.groupSize,
                        bits: quantization.bits,
                        keyMapper: keyMapper,
                        mapper: mapper
                    )
                } else {
                    try HFSafetensorsWeightsLoader.applyWeights(
                        url: singleFileURL,
                        to: encoder,
                        verify: .noUnusedKeys,
                        mapper: mapper
                    )
                }
            }
            return
        }

        // Sharded weights
        let files = try FileManager.default.contentsOfDirectoryResolvingSymlinks(
            at: url,
            includingPropertiesForKeys: nil
        )
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !files.isEmpty else {
            throw NSError(domain: "Flux2KleinGeneratoriOS", code: 1, userInfo: [NSLocalizedDescriptionKey: "No text encoder weights found at \(url.path)"])
        }

        do {
            let firstWeights = try MLX.loadArrays(url: files[0])
            let isQuantized = HFSafetensorsWeightsLoader.isQuantized(firstWeights)

            if isQuantized {
                guard let quantization else {
                    throw ModelWeightsLoader.LoaderError.invalidQuantizationMetadata(
                        "Quantized text encoder weights detected, but quantization params were not provided (missing \(MereRunModelManifest.filename))."
                    )
                }
                var allWeights: [String: MLXArray] = [:]
                for (key, value) in firstWeights {
                    for (mappedKey, mappedValue) in mapper(key, value) {
                        allWeights[mappedKey] = mappedValue
                    }
                }
                for file in files.dropFirst() {
                    let weights = try MLX.loadArrays(url: file)
                    for (key, value) in weights {
                        for (mappedKey, mappedValue) in mapper(key, value) {
                            allWeights[mappedKey] = mappedValue
                        }
                    }
                }
                try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                    allWeights,
                    to: encoder,
                    groupSize: quantization.groupSize,
                    bits: quantization.bits
                )
            } else {
                for file in files {
                    try HFSafetensorsWeightsLoader.applyWeights(
                        url: file,
                        to: encoder,
                        verify: .noUnusedKeys,
                        mapper: mapper
                    )
                }
            }
        }
    }


}
