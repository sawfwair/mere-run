import Foundation
import MLX
import MLXNN
import AudioCore
import MereRunCore

/// Owns model resolution and weight loading for the TTS stack.
/// This file prepares the runtime but does not generate audio.
extension Qwen3TTSGenerator {
    func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (TTSProgress) -> Void)?
    ) async throws -> URL {
        do {
            let resolved = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: modelPath ?? modelId,
                defaultModelID: modelId,
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(TTSProgress(stage: .loadingModel, message: "Downloading model... \(percent)%"))
                    case .extracting:
                        progressHandler?(TTSProgress(stage: .loadingModel, message: "Extracting model..."))
                    }
                }
            )
            return resolved.url
        } catch let error as ManagedModelResolver.ResolverError {
            throw Qwen3TTSError.downloadFailed(error.localizedDescription)
        }
    }

    func mapModelLoaderError(_ error: PretrainedModelLoader.LoadError) -> Qwen3TTSError {
        switch error {
        case .unsupportedModelId(let modelId):
            return .unsupportedModelId(modelId)
        case .missingFiles(let files):
            return .missingFiles(files)
        case .downloadFailed(let message):
            return .downloadFailed(message)
        case .extractionFailed:
            return .extractionFailed
        }
    }

    func loadModels(
        from rootURL: URL,
        progressHandler: (@Sendable (TTSProgress) -> Void)?
    ) async throws {
        let resources = Qwen3TTSResources(rootURL: rootURL)
        let missingFiles = resources.validate()
        guard missingFiles.isEmpty else {
            throw Qwen3TTSError.missingFiles(missingFiles.map { $0.lastPathComponent })
        }

        let config = try Qwen3TTSModelConfig.load(from: resources.configURL)
        let talker = Qwen3TTSTalkerForConditionalGeneration(config: config.talkerConfig)

        progressHandler?(TTSProgress(stage: .loadingModel, message: "Loading model weights..."))
        try loadTalkerWeights(resources: resources, talker: talker, quantization: config.quantization)

        progressHandler?(TTSProgress(stage: .loadingModel, message: "Loading tokenizer..."))
        let tokenizer = try Qwen3TTSTokenizer.load(from: rootURL)

        progressHandler?(TTSProgress(stage: .loadingModel, message: "Loading speech tokenizer..."))
        let speechTokenizer = try loadSpeechTokenizer(resources: resources)

        let speakerEncoder = try loadSpeakerEncoder(
            resources: resources,
            config: config,
            progressHandler: progressHandler
        )

        self.talker = talker
        self.speechTokenizer = speechTokenizer
        self.speakerEncoder = speakerEncoder
        self.tokenizer = tokenizer
        self.modelConfig = config
        self.loadedModelPath = rootURL.path
    }

    func loadTalkerWeights(
        resources: Qwen3TTSResources,
        talker: Qwen3TTSTalkerForConditionalGeneration,
        quantization: Qwen3TTSQuantizationConfig?
    ) throws {
        let groupSize = quantization?.groupSize ?? 64
        let bits = quantization?.bits ?? 4
        let indexURL = resources.modelIndexURL
        let singleURL = resources.modelWeightsURL

        if FileManager.default.fileExists(atPath: indexURL.path) {
            do {
                try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                    indexURL: indexURL,
                    to: talker,
                    groupSize: groupSize,
                    bits: bits,
                    keyMapper: mapTalkerWeightKey
                )
            } catch HFSafetensorsWeightsLoader.LoaderError.notQuantizedWeights {
                try HFSafetensorsWeightsLoader.applyShardedWeights(
                    indexURL: indexURL,
                    to: talker,
                    dtype: .bfloat16,
                    verify: .none,
                    mapper: { key, value in
                        guard key.hasPrefix("talker.") else { return [] }
                        return [(mapTalkerWeightKey(key), value)]
                    }
                )
            }
            return
        }

        if FileManager.default.fileExists(atPath: singleURL.path) {
            var arrays = try MLX.loadArrays(url: singleURL).filter { $0.key.hasPrefix("talker.") }
            if HFSafetensorsWeightsLoader.isQuantized(arrays) {
                try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                    arrays,
                    to: talker,
                    groupSize: groupSize,
                    bits: bits,
                    keyMapper: mapTalkerWeightKey
                )
            } else {
                try HFSafetensorsWeightsLoader.applyWeights(
                    url: singleURL,
                    to: talker,
                    dtype: .bfloat16,
                    verify: .none,
                    mapper: { key, value in
                        guard key.hasPrefix("talker.") else { return [] }
                        return [(mapTalkerWeightKey(key), value)]
                    }
                )
            }
            arrays.removeAll()
            return
        }

        throw Qwen3TTSError.weightsNotFound(indexURL)
    }

    func loadSpeechTokenizer(resources: Qwen3TTSResources) throws -> Qwen3TTSSpeechTokenizer {
        let data = try Data(contentsOf: resources.speechTokenizerConfigURL)
        let config = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: data)
        let tokenizer = Qwen3TTSSpeechTokenizer(config: config)

        var weights: [String: MLXArray] = [:]
        for url in resources.speechTokenizerWeightsURLs {
            let arrays = try MLX.loadArrays(url: url)
            for (key, value) in arrays {
                weights[key] = value
            }
        }
        if weights.isEmpty {
            throw Qwen3TTSError.weightsNotFound(resources.speechTokenizerDirURL)
        }

        let sanitized = Qwen3TTSSpeechTokenizer.sanitize(weights, config: config)
        try tokenizer.update(parameters: ModuleParameters.unflattened(sanitized.map { ($0.key, $0.value) }), verify: .none)
        return tokenizer
    }

    func loadSpeakerEncoder(
        resources: Qwen3TTSResources,
        config: Qwen3TTSModelConfig,
        progressHandler: (@Sendable (TTSProgress) -> Void)?
    ) throws -> Qwen3TTSSpeakerEncoder? {
        guard let speakerConfig = config.speakerEncoderConfig else {
            return nil
        }

        progressHandler?(TTSProgress(stage: .loadingModel, message: "Loading speaker encoder..."))
        let encoder = Qwen3TTSSpeakerEncoder(config: speakerConfig)
        var arrays: [String: MLXArray] = [:]

        for url in resources.speakerEncoderWeightsURLs {
            let loaded = try MLX.loadArrays(url: url)
            for (key, value) in loaded {
                arrays[key] = value
            }
        }

        if arrays.isEmpty {
            if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
                let loaded = try HFSafetensorsWeightsLoader.loadShardedArrays(indexURL: resources.modelIndexURL)
                for (key, value) in loaded where key.hasPrefix("speaker_encoder.") {
                    arrays[key] = value
                }
            } else if FileManager.default.fileExists(atPath: resources.modelWeightsURL.path) {
                let loaded = try MLX.loadArrays(url: resources.modelWeightsURL)
                for (key, value) in loaded where key.hasPrefix("speaker_encoder.") {
                    arrays[key] = value
                }
            }
        }

        let sanitized = Qwen3TTSSpeakerEncoder.sanitize(arrays)
        guard !sanitized.isEmpty else {
            return nil
        }

        try encoder.update(parameters: ModuleParameters.unflattened(sanitized.map { ($0.key, $0.value) }), verify: .none)
        return encoder
    }
}
