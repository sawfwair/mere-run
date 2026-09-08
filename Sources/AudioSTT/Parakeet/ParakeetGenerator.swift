import Foundation
import MLX
import MLXNN
import AudioCore
import AudioCodecs
import MereRunCore

public actor ParakeetGenerator: ASRGenerator {
    var model: (any ParakeetDecodingModel)?
    var audioPreprocessor: ParakeetAudioPreprocessor?
    var modelConfig: ParakeetModelConfig?
    var loadedModelPath: String?

    let modelId: String
    let executionProvider: ParakeetExecutionProvider

    static let debugEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_ASR_DEBUG"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }()

    public init(
        modelId: String = ParakeetResources.defaultModelId,
        executionProvider: ParakeetExecutionProvider = .mlx
    ) {
        self.modelId = modelId
        self.executionProvider = executionProvider
    }

    init(
        preparedModel: any ParakeetDecodingModel,
        audioPreprocessor: ParakeetAudioPreprocessor,
        modelConfig: ParakeetModelConfig,
        executionProvider: ParakeetExecutionProvider = .mlx
    ) {
        self.modelId = ParakeetResources.defaultModelId
        self.executionProvider = executionProvider
        self.model = preparedModel
        self.audioPreprocessor = audioPreprocessor
        self.modelConfig = modelConfig
        self.loadedModelPath = "<prepared>"
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

        progressHandler?(ASRProgress(stage: .loadingAudio, message: "Loading audio..."))
        let audio = try AudioReader.readAudio(from: request.audioURL)
        return try await transcribePrepared(
            samples: audio,
            language: request.language,
            progressHandler: progressHandler
        )
    }

    public func transcribe(
        samples: [Float],
        language: String? = nil,
        modelPath: String? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        let root = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)

        if loadedModelPath != root.path {
            progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading Parakeet model..."))
            try await loadModel(from: root, progressHandler: progressHandler)
        }

        return try await transcribePrepared(
            samples: samples,
            language: language,
            progressHandler: progressHandler
        )
    }

    public func transcribePrepared(
        samples: [Float],
        language: String? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        try await transcribePreparedMeasured(
            samples: samples,
            language: language,
            progressHandler: progressHandler
        ).result
    }

    public func transcribePreparedMeasured(
        samples: [Float],
        language: String? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ParakeetMeasuredTranscription {
        try await Stream.withNewDefaultStream(isolation: self) {
            // Keep this closure genuinely asynchronous under -O. MLX's async overload
            // installs task-local CPU and GPU streams that survive executor hops; without
            // a suspension the optimizer can collapse the body onto the caller thread and
            // `.gpu` operations fall back to MLX's missing thread-local default stream.
            await Task.yield()
            guard let model, let audioPreprocessor, let modelConfig else {
                throw ParakeetError.modelNotLoaded
            }

            return try decodeMeasured(
                samples: samples,
                language: language,
                model: model,
                audioPreprocessor: audioPreprocessor,
                modelConfig: modelConfig,
                progressHandler: progressHandler
            )
        }
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
}
