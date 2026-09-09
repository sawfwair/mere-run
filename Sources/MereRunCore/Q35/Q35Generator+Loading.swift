import Foundation
import MediaIO
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

extension Q35Generator {
    func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let normalizedRoot = Q35Resources.normalizedRootURL(rootURL)
        if loadedModelPath == normalizedRoot.path, model != nil, tokenizerAndTemplate != nil {
            return
        }
        resetPrefixKVCache()

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen-family config"))
        let configData = try Data(contentsOf: normalizedRoot.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Q35Config.self, from: configData)
        let generationConfigURL = normalizedRoot.appendingPathComponent("generation_config.json")
        let generationEOSTokenIds: [Int]
        if FileManager.default.fileExists(atPath: generationConfigURL.path) {
            let generationConfigData = try Data(contentsOf: generationConfigURL)
            generationEOSTokenIds = try JSONDecoder()
                .decode(Q35GenerationConfig.self, from: generationConfigData)
                .eosTokenIds
        } else {
            generationEOSTokenIds = []
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen-family tokenizer"))
        let tokenizer = try Q35TokenizerAndTemplate.load(
            from: normalizedRoot,
            maxLengthOverride: config.textConfig.maxPositionEmbeddings
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen-family weights"))
        let q35Model = Q35Model(
            config: config,
            asynchronousDecodeBlocks: Q35RuntimeTuning.isEnabled(.asynchronousDecode, modelID: modelId)
        )
        let resources = Q35Resources(rootURL: normalizedRoot)

        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 4

        try loadTextWeights(
            into: q35Model,
            from: resources,
            groupSize: groupSize,
            bits: bits
        )
        if config.textConfig.isQwen4Exp {
            progressHandler?(ChatProgress(
                stage: .loadingModel,
                message: "Loading Qwen4Exp n-gram embedding shards"
            ))
            try loadQ38NGramEmbeddings(
                into: q35Model,
                from: resources,
                progressHandler: progressHandler
            )
        }
        #if DEBUG
        try checkpointTransformForTesting?(q35Model)
        #endif
        if Q35FusedSwitchGLUPolicy.enabled, config.textConfig.usesMoE {
            progressHandler?(ChatProgress(
                stage: .loadingModel,
                message: "Preparing Qwen-family fused MoE weights"
            ))
            _ = q35Model.prepareFusedSwitchGLU()
        }

        let primaryResources = Q35Resources(rootURL: normalizedRoot)
        let visionConfigAndResources: (Q35Config, Q35Resources)?
        if config.visionConfig != nil {
            visionConfigAndResources = (config, primaryResources)
        } else if modelId == Q35Resources.q38TwentySevenB4BitModelId {
            let companionResources = primaryResources.q38VisionComponentResources
            let companionConfigData = try Data(contentsOf: companionResources.configURL)
            let companionConfig = try JSONDecoder().decode(Q35Config.self, from: companionConfigData)
            guard companionConfig.visionConfig != nil else {
                throw Q35Error.generationFailed("Qwen3.8 4-bit vision companion does not include a vision config.")
            }
            visionConfigAndResources = (companionConfig, companionResources)
        } else if modelId == Q35Resources.ornith35BMLX4BitModelId {
            let companionResources = primaryResources.ornithVisionComponentResources
            let companionConfigData = try Data(contentsOf: companionResources.configURL)
            let companionConfig = try JSONDecoder().decode(Q35Config.self, from: companionConfigData)
            guard companionConfig.visionConfig != nil else {
                throw Q35Error.generationFailed("Ornith 4-bit vision companion does not include a vision config.")
            }
            visionConfigAndResources = (companionConfig, companionResources)
        } else {
            visionConfigAndResources = nil
        }
        let tower = visionConfigAndResources.map { Q35VisionTower(config: $0.0) }
        // Load the MTP draft head whenever it ships with the model and isn't
        // explicitly disabled. Whether speculation is actually USED is decided
        // by the model-specific policy in Self.shouldSpeculate. BF16 Qwen3.8
        // remains opt-in, Q4 Qwen3.8 and Ornith use their short-context
        // paths, and Qwen3.6 hybrid MoE retains the measured long-context threshold.
        let mtpPolicy = ProcessInfo.processInfo.environment["MERERUN_Q35_MTP_SPECULATION"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let mtpExplicitlyDisabled = mtpPolicy == "0" || mtpPolicy == "false" || mtpPolicy == "no"
        let mtpExplicitlyEnabled = mtpPolicy == "1" || mtpPolicy == "true"
            || mtpPolicy == "yes" || mtpPolicy == "on"
        let q38Q4 = modelId == Q35Resources.q38TwentySevenB4BitModelId
        let shouldLoadMTP = !mtpExplicitlyDisabled
            && (config.textConfig.usesMoE || q38Q4 || mtpExplicitlyEnabled)
        let loadedMTP: (any Q35MTPDraftModel)?
        let ornithMTPCompanionRoot = Q35Resources.isOrnith35BMLXModelId(modelId)
            ? ManagedModelResolver.resolveInstalledModel(id: Q35Resources.ornith35BMTPModelId)
            : nil
        if shouldLoadMTP,
           let mtpResources = Self.mtpResources(
               primary: resources,
               companionRootURL: ornithMTPCompanionRoot
           ) {
            if config.textConfig.isQwen4Exp {
                guard config.textConfig.mtp?.numHiddenLayers == 1 else {
                    throw Q35Error.generationFailed(
                        "Qwen4Exp MTP requires the published one-layer draft configuration."
                    )
                }
                progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen4Exp MTP weights"))
                let mtp = Q38MTPModel(config: config)
                try loadQ38MTPWeights(
                    into: mtp,
                    from: mtpResources,
                    groupSize: groupSize,
                    bits: bits
                )
                if Q35FusedSwitchGLUPolicy.enabled {
                    progressHandler?(ChatProgress(
                        stage: .loadingModel,
                        message: "Preparing Qwen4Exp MTP fused MoE weights"
                    ))
                    _ = mtp.prepareFusedSwitchGLU()
                }
                loadedMTP = mtp
            } else {
                progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen-family MTP weights"))
                let mtp = Q35MTPModel(config: config)
                try loadMTPWeights(
                    into: mtp,
                    baseModel: q35Model,
                    from: mtpResources,
                    groupSize: groupSize,
                    bits: bits
                )
                loadedMTP = mtp
            }
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Preparing Qwen-family MTP drafting"))
            q35Model.prepareGreedyMTPDrafting()
        } else {
            loadedMTP = nil
        }

        model = q35Model
        tokenizerAndTemplate = tokenizer
        visionTower = tower
        mtpModel = loadedMTP
        loadedConfig = config
        loadedGenerationEOSTokenIds = generationEOSTokenIds
        loadedResources = resources
        loadedVisionResources = visionConfigAndResources?.1
        loadedModelPath = normalizedRoot.path
    }
    func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        guard let profile = Q35Resources.profile(for: modelId) else {
            throw Q35Error.unsupportedModelId(modelId)
        }

        do {
            let root = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: modelPath ?? modelId,
                defaultModelID: profile.modelId,
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(ChatProgress(stage: .loadingModel, message: "Downloading model... \(percent)%"))
                    case .extracting:
                        progressHandler?(ChatProgress(stage: .loadingModel, message: "Extracting model..."))
                    }
                }
            )
            return Q35Resources.normalizedRootURL(root.url)
        } catch let error as ManagedModelResolver.ResolverError {
            throw Q35Error.downloadFailed(error.localizedDescription)
        }
    }

    private func mapLoaderError(_ error: PretrainedModelLoader.LoadError) -> Q35Error {
        switch error {
        case .unsupportedModelId(let modelId):
            return .unsupportedModelId(modelId)
        case .missingFiles(let files):
            return .missingFiles(files)
        case .downloadFailed(let message):
            return .downloadFailed(message)
        }
    }

    func makeLayerCaches(
        config: Q35Config,
        kvCacheMode: RuntimeKVCacheMode = .default
    ) -> [Q35LayerCache?] {
        let text = config.textConfig
        let mlpOnly = Set(text.mlpOnlyLayers)
        return (0..<text.numHiddenLayers).map { layerIndex in
            if mlpOnly.contains(layerIndex) {
                return nil
            }
            let layerType = layerIndex < text.layerTypes.count ? text.layerTypes[layerIndex] : "linear_attention"
            if layerType == "full_attention" {
                let attention: KVCache
                if kvCacheMode == .affine4 || kvCacheMode == .affine8 {
                    attention = AffineQuantizedKVCache(
                        groupSize: Self.affineKVGroupSize(headDimension: text.headDim),
                        bits: kvCacheMode == .affine4 ? 4 : 8,
                        step: 256
                    )
                } else {
                    attention = KVCacheSimple(step: 256)
                }
                return .full(text.isQwen4Exp ? Q38QSACache(attention: attention) : attention)
            }
            return .linear(Q35LinearCache())
        }
    }

    func cacheMode(for caches: [Q35LayerCache?]) -> RuntimeKVCacheMode {
        caches.contains { entry in
            guard case .full(let cache)? = entry else { return false }
            let main = (cache as? Q38QSACache)?.attention ?? cache
            guard let affine = main as? AffineQuantizedKVCache else { return false }
            return affine.bitWidth == 4
        } ? .affine4 : caches.contains { entry in
            guard case .full(let cache)? = entry else { return false }
            return ((cache as? Q38QSACache)?.attention ?? cache) is AffineQuantizedKVCache
        } ? .affine8 : .default
    }

    private static func affineKVGroupSize(headDimension: Int) -> Int {
        [64, 32, 16, 8].first { headDimension % $0 == 0 } ?? 1
    }
}
