import Foundation
import MLX

extension LagunaGenerator {
    func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let requestedDFlashPath = configuredDFlashPath.map {
            URL(fileURLWithPath: $0).standardizedFileURL.path
        }
        if loadedModelPath == rootURL.path,
           loadedDFlashPath == requestedDFlashPath,
           model != nil,
           tokenizerAndTemplate != nil {
            return
        }

        let missingFiles = LagunaResources.validate(rootURL: rootURL)
        guard missingFiles.isEmpty else {
            throw LagunaError.missingFiles(missingFiles)
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Laguna config"))
        let configData = try Data(contentsOf: rootURL.appending(path: "config.json"))
        let config = try JSONDecoder().decode(LagunaConfig.self, from: configData)
        let weightsIndexURL = rootURL.appending(path: "model.safetensors.index.json")
        let weightsIndex = try JSONDecoder().decode(
            HFSafetensorsIndex.self,
            from: Data(contentsOf: weightsIndexURL)
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Laguna tokenizer"))
        let tokenizer = try await LagunaTokenizerAndTemplate.load(
            from: rootURL,
            maxLength: config.maxPositionEmbeddings
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Laguna weights"))
        let model = LagunaCausalLM(
            config: config,
            quantizedSharedExperts: LagunaResources.hasQuantizedSharedExperts(weightsIndex)
        )
        try HFSafetensorsWeightsLoader.applyShardedWeights(
            indexURL: weightsIndexURL,
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
        let runtimeAccelerationArrays = model.prepareRuntimeAcceleration()
        if !runtimeAccelerationArrays.isEmpty {
            MLX.eval(runtimeAccelerationArrays)
        }

        self.model = model
        self.tokenizerAndTemplate = tokenizer
        self.config = config
        self.loadedModelPath = rootURL.path
        self.loadedTextLoRASignature = nil

        if let configuredDFlashPath {
            let dflashRootURL = URL(fileURLWithPath: configuredDFlashPath)
                .standardizedFileURL
            let missingDFlashFiles = LagunaResources.validateDFlash(
                rootURL: dflashRootURL
            )
            guard missingDFlashFiles.isEmpty else {
                throw LagunaError.missingFiles(
                    missingDFlashFiles.map { "DFlash/\($0)" }
                )
            }
            progressHandler?(ChatProgress(
                stage: .loadingModel,
                message: "Loading Laguna DFlash config"
            ))
            let dflashConfig = try JSONDecoder().decode(
                LagunaDFlashConfig.self,
                from: Data(contentsOf: dflashRootURL.appending(path: "config.json"))
            )
            try validateDFlashCompatibility(
                target: config,
                dflash: dflashConfig
            )
            progressHandler?(ChatProgress(
                stage: .loadingModel,
                message: "Loading Laguna DFlash weights"
            ))
            let dflashModel = LagunaDFlashModel(config: dflashConfig)
            try HFSafetensorsWeightsLoader.applyWeights(
                url: dflashRootURL.appending(path: "model.safetensors"),
                to: dflashModel,
                dtype: nil,
                verify: .shapeMismatch
            )
            self.dflashModel = dflashModel
            self.dflashConfig = dflashConfig
            self.loadedDFlashPath = dflashRootURL.path
        } else {
            dflashModel = nil
            dflashConfig = nil
            loadedDFlashPath = nil
        }
    }

    func applyTextLoRAIfNeeded(
        _ lora: LoRA?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        guard let lora else {
            loadedTextLoRASignature = nil
            return
        }
        let signature = Self.loraSignature(lora)
        guard loadedTextLoRASignature != signature else { return }
        guard let model else {
            throw LagunaError.modelNotLoaded
        }
        progressHandler?(ChatProgress(
            stage: .loadingModel,
            message: "Loading Laguna text LoRA"
        ))
        _ = try await LagunaTextLoRAAdapter.apply(lora, to: model)
        loadedTextLoRASignature = signature
    }

    static func loraSignature(_ lora: LoRA?) -> String? {
        guard let lora else { return nil }
        switch lora {
        case .local(let path, let scale):
            return "local:\(URL(fileURLWithPath: path).standardizedFileURL.path):\(scale)"
        case .remote(let reference, let scale):
            return "remote:\(reference):\(scale)"
        }
    }

    func warmUp(
        model: LagunaCausalLM,
        dflash: LagunaDFlashModel?
    ) {
        let token = MLXArray([Int32(0)]).reshaped(1, 1)
        let captureLayerIndices = dflash.map {
            Set($0.config.dflash.targetLayerIDs)
        } ?? []
        let targetCache = model.makeCache()
        let output = model.forward(
            token,
            cache: targetCache,
            captureLayerIndices: captureLayerIndices,
            lastPositionOnly: true
        )
        guard let dflash else {
            MLX.eval(output.logits)
            targetCache.forEach { $0.evaluateStorage() }
            return
        }

        let draftCache = dflash.makeCache()
        dflash.appendTargetContext(
            dflash.combineTargetHiddenStates(output.capturedHiddenStates),
            cache: draftCache
        )
        let draftLogits = dflash.draftLogits(
            anchorTokens: token,
            speculativeTokenCount: 1,
            cache: draftCache,
            target: model
        )
        MLX.eval(output.logits, draftLogits)
        targetCache.forEach { $0.evaluateStorage() }
        draftCache.forEach { $0.evaluateStorage() }
    }

    func validateDFlashCompatibility(
        target: LagunaConfig,
        dflash: LagunaDFlashConfig
    ) throws {
        guard dflash.vocabSize == target.vocabSize else {
            throw LagunaError.dflashIncompatible(
                "draft vocabulary \(dflash.vocabSize) does not match target \(target.vocabSize)."
            )
        }
        guard dflash.hiddenSize == target.hiddenSize else {
            throw LagunaError.dflashIncompatible(
                "draft hidden size \(dflash.hiddenSize) does not match target \(target.hiddenSize)."
            )
        }
        guard dflash.dflash.numTargetLayers == target.numHiddenLayers else {
            throw LagunaError.dflashIncompatible(
                "draft expects \(dflash.dflash.numTargetLayers) target layers, found \(target.numHiddenLayers)."
            )
        }
        guard dflash.dflash.targetLayerIDs.allSatisfy({
            target.modelType == "laguna" && target.numHiddenLayers > $0
        }) else {
            throw LagunaError.dflashIncompatible(
                "target_layer_ids reference layers outside the target model."
            )
        }
    }

}
