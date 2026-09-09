import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioCore
import AudioCodecs
import MereRunCore

extension Qwen3ASRGenerator {
    // MARK: - Model Resolution

    func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> URL {
        do {
            let resolved = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: modelPath ?? modelId,
                defaultModelID: Qwen3ASRResources.defaultModelId,
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(
                            ASRProgress(stage: .loadingModel, message: "Downloading model... \(percent)%")
                        )
                    case .extracting:
                        progressHandler?(
                            ASRProgress(stage: .loadingModel, message: "Extracting model...")
                        )
                    }
                }
            )
            return resolved.url
        } catch let error as ManagedModelResolver.ResolverError {
            throw Qwen3ASRError.downloadFailed(error.localizedDescription)
        }
    }

    private func mapModelLoaderError(_ error: PretrainedModelLoader.LoadError) -> Qwen3ASRError {
        switch error {
        case .unsupportedModelId(let modelId):
            return .unsupportedModelId(modelId)
        case .missingFiles(let files):
            return .missingFiles(files)
        case .downloadFailed(let message):
            return .downloadFailed(message)
        }
    }

    // MARK: - Model Loading

    func loadModels(
        from rootURL: URL,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws {
        let resources = Qwen3ASRResources(rootURL: rootURL)
        let missingFiles = resources.validate()
        guard missingFiles.isEmpty else {
            throw Qwen3ASRError.missingFiles(missingFiles.map { $0.lastPathComponent })
        }

        let config = try Qwen3ASRModelConfig.load(from: resources.configURL)
        let thinker = Qwen3ASRThinker(config: config)

        progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading model weights..."))
        try loadModelWeights(resources: resources, thinker: thinker)

        progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading tokenizer..."))
        let tokenizer = try Qwen3ASRTokenizer.load(from: rootURL, config: config)

        let melExtractor = MelSpectrogram()

        self.thinker = thinker
        self.tokenizer = tokenizer
        self.melExtractor = melExtractor
        self.modelConfig = config
        self.loadedModelPath = rootURL.path
    }

    private func loadModelWeights(
        resources: Qwen3ASRResources,
        thinker: Qwen3ASRThinker
    ) throws {
        let indexURL = resources.modelIndexURL
        let singleURL = resources.modelWeightsURL
        let fm = FileManager.default

        let arrays: [String: MLXArray]
        if fm.fileExists(atPath: indexURL.path) {
            arrays = try HFSafetensorsWeightsLoader.loadShardedArrays(indexURL: indexURL)
        } else if fm.fileExists(atPath: singleURL.path) {
            arrays = try MLX.loadArrays(url: singleURL)
        } else {
            throw Qwen3ASRError.weightsNotFound(indexURL)
        }

        let mapped = mapASRWeights(
            arrays,
            tieWordEmbeddings: thinker.config.textConfig.tieWordEmbeddings
        )

        if HFSafetensorsWeightsLoader.isQuantized(mapped) {
            let bits = thinker.config.quantizationBits ?? 8
            let groupSize = thinker.config.quantizationGroupSize ?? 64
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                mapped,
                to: thinker,
                groupSize: groupSize,
                bits: bits
            )
        } else {
            var updates: [(String, MLXArray)] = []
            updates.reserveCapacity(mapped.count)
            for (key, value) in mapped {
                let casted = HFSafetensorsWeightsLoader.castIfNeeded(value, dtype: .bfloat16)
                updates.append((key, casted))
            }
            try thinker.update(parameters: ModuleParameters.unflattened(updates), verify: .none)
        }
    }
}

// MARK: - Weight Key Mapping

private func mapASRWeightKey(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
    var mapped = key

    // Drop optional thinker prefix
    if mapped.hasPrefix("thinker.") {
        mapped = String(mapped.dropFirst("thinker.".count))
    }

    // Only accept known roots
    guard mapped.hasPrefix("audio_tower.")
        || mapped.hasPrefix("model.")
        || mapped.hasPrefix("lm_head.") else {
        return []
    }

    // Conv2d weights: convert only if stored as OIHW
    if mapped.contains("conv2d") && mapped.hasSuffix(".weight") {
        return [(mapped, convertConv2DWeightIfNeeded(value))]
    }

    return [(mapped, value)]
}

private func mapASRWeights(
    _ arrays: [String: MLXArray],
    tieWordEmbeddings: Bool
) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]
    mapped.reserveCapacity(arrays.count)

    for (key, value) in arrays {
        var mappedKey = key
        if mappedKey.hasPrefix("thinker.") {
            mappedKey = String(mappedKey.dropFirst("thinker.".count))
        }

        if !(mappedKey.hasPrefix("audio_tower.")
            || mappedKey.hasPrefix("model.")
            || mappedKey.hasPrefix("lm_head.")) {
            continue
        }

        if tieWordEmbeddings, mappedKey.hasPrefix("lm_head.") {
            continue
        }

        let mappedValue: MLXArray
        if mappedKey.contains("conv2d") && mappedKey.hasSuffix(".weight") {
            mappedValue = convertConv2DWeightIfNeeded(value)
        } else {
            mappedValue = value
        }

        mapped[mappedKey] = mappedValue
    }

    return mapped
}

private func convertConv2DWeightIfNeeded(_ value: MLXArray) -> MLXArray {
    guard value.ndim == 4 else { return value }
    let s = value.shape
    // Qwen3-ASR conv kernels are 3x3. Detect layout by which dims are 3.
    if s[1] == 3 && s[2] == 3 {
        // Already OHWI (out, kH, kW, in)
        return value
    }
    if s[2] == 3 && s[3] == 3 {
        // OIHW -> OHWI
        return HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(value)
    }
    return value
}
