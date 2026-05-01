import Foundation
import MLX
import MLXNN
import AudioCore
import AudioCodecs
import MereRunCore

public actor ParakeetGenerator: ASRGenerator {
    private var model: (any ParakeetDecodingModel)?
    private var audioPreprocessor: ParakeetAudioPreprocessor?
    private var modelConfig: ParakeetModelConfig?
    private var loadedModelPath: String?

    private let modelId: String

    private static let debugEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_ASR_DEBUG"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }()

    public init(modelId: String = ParakeetResources.defaultModelId) {
        self.modelId = modelId
    }

    public func transcribe(
        _ request: ASRRequest,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        try await transcribe(request, modelPath: nil, progressHandler: progressHandler)
    }

    public func transcribe(
        _ request: ASRRequest,
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        let root = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)

        if loadedModelPath != root.path {
            progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading Parakeet model..."))
            try await loadModel(from: root, progressHandler: progressHandler)
        }

        guard let model, let audioPreprocessor, let modelConfig else {
            throw ParakeetError.modelNotLoaded
        }

        progressHandler?(ASRProgress(stage: .loadingAudio, message: "Loading audio..."))
        let audio = try AudioReader.readAudio(from: request.audioURL)
        let audioDuration = TimeInterval(audio.count) / TimeInterval(modelConfig.preprocessor.sampleRate)

        progressHandler?(ASRProgress(stage: .extractingFeatures, message: "Extracting log-mel features..."))
        let mel = audioPreprocessor.logMelSpectrogram(from: audio)
        MLX.eval(mel)

        if Self.debugEnabled {
            let melMin = MLX.min(mel).item(Float.self)
            let melMax = MLX.max(mel).item(Float.self)
            let melMean = MLX.mean(mel).item(Float.self)
            FileHandle.standardError.write(
                Data("[ASR DEBUG] parakeet_mel shape=\(mel.shape) mean=\(melMean) min=\(melMin) max=\(melMax)\n".utf8)
            )
        }

        progressHandler?(ASRProgress(stage: .transcribing, message: "Transcribing with Parakeet..."))
        let decoded = model.decode(mel)
        let first = decoded.first ?? ParakeetAlignedResult(text: "", sentences: [])

        var flattened: [ParakeetAlignedToken] = []
        flattened.reserveCapacity(first.sentences.reduce(0) { $0 + $1.tokens.count })
        for sentence in first.sentences {
            flattened.append(contentsOf: sentence.tokens)
        }

        if Self.debugEnabled {
            let message = "[ASR DEBUG] parakeet_backend=\(modelConfig.variant.rawValue) tokens=\(flattened.count)\n"
            FileHandle.standardError.write(Data(message.utf8))
            let preview = flattened.prefix(24).map { "\($0.id):\($0.text)" }.joined(separator: " | ")
            FileHandle.standardError.write(Data("[ASR DEBUG] parakeet_tokens_preview \(preview)\n".utf8))
        }

        Memory.clearCache()

        return ASRResult(
            text: first.text,
            language: request.language,
            duration: audioDuration,
            tokenAlignments: ParakeetAlignment.toASRTokenAlignments(flattened),
            sentenceAlignments: ParakeetAlignment.toASRSentenceAlignments(first.sentences)
        )
    }

    public func prepare(
        modelPath: String? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws {
        let root = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
        if loadedModelPath != root.path {
            progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading Parakeet model..."))
            try await loadModel(from: root, progressHandler: progressHandler)
        }
    }

    public func unload() {
        model = nil
        audioPreprocessor = nil
        modelConfig = nil
        loadedModelPath = nil
        Memory.clearCache()
    }

    public func supportedLanguageCodes(modelPath: String? = nil) async throws -> Set<String> {
        let root = try await resolveModelRoot(modelPath: modelPath, progressHandler: nil)
        let resources = ParakeetResources(rootURL: root)
        let config = try ParakeetModelConfig.load(from: resources.configURL)
        return config.supportedLanguageCodes
    }

    private func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> URL {
        do {
            let resolved = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: modelPath ?? modelId,
                defaultModelID: ParakeetResources.defaultModelId,
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
            throw ParakeetError.downloadFailed(error.localizedDescription)
        }
    }

    private func mapModelLoaderError(_ error: PretrainedModelLoader.LoadError) -> ParakeetError {
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

    private func loadModel(
        from rootURL: URL,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws {
        let resources = ParakeetResources(rootURL: rootURL)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw ParakeetError.missingFiles(missing.map(\.lastPathComponent))
        }

        let config = try ParakeetModelConfig.load(from: resources.configURL)
        let model = ParakeetModelFactory.build(config: config)

        guard let module = model as? Module else {
            throw ParakeetError.modelNotLoaded
        }

        progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading Parakeet weights..."))

        let quantization: ModelWeightsLoader.QuantizationParams?
        if let bits = config.quantizationBits, let group = config.quantizationGroupSize {
            quantization = .init(bits: bits, groupSize: group)
        } else {
            quantization = nil
        }

        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.modelIndexURL,
            singleURL: resources.modelWeightsURL,
            to: module,
            dtype: .bfloat16,
            verify: .noUnusedKeys,
            quantization: quantization
        )

        // Parakeet uses BatchNorm/Dropout in the encoder stack; keep inference in eval mode.
        module.train(false)

        MLX.eval(module.parameters())

        self.model = model
        self.modelConfig = config
        self.audioPreprocessor = try ParakeetAudioPreprocessor(config: config.preprocessor)
        self.loadedModelPath = rootURL.path
    }
}

public enum ParakeetError: LocalizedError {
    case modelNotLoaded
    case unsupportedModelId(String)
    case missingFiles([String])
    case downloadFailed(String)
    case extractionFailed

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Parakeet model is not loaded."
        case .unsupportedModelId(let modelId):
            return "Unsupported model id: \(modelId)"
        case .missingFiles(let files):
            return "Parakeet model files missing: \(files.joined(separator: ", "))"
        case .downloadFailed(let message):
            return message
        case .extractionFailed:
            return "Failed to extract model archive"
        }
    }
}
