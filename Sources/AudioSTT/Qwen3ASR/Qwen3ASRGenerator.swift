import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioCore
import AudioCodecs
import MereRunCore

// MARK: - Qwen3 ASR Generator

/// Native Swift implementation for Qwen3-ASR speech recognition
public actor Qwen3ASRGenerator: ASRGenerator {

    // MARK: - Actor State

    var thinker: Qwen3ASRThinker?
    var tokenizer: Qwen3ASRTokenizer?
    var melExtractor: MelSpectrogram?
    var modelConfig: Qwen3ASRModelConfig?
    var loadedModelPath: String?

    let modelId: String
    static let debugEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_ASR_DEBUG"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }()

    /// Depth-1 pipelined decode (default on): the sampled token feeds the
    /// next forward as a GPU array and the previous token is read back while
    /// the current step executes. MERERUN_STT_PIPELINED_DECODE=0 restores
    /// the legacy two-syncs-per-token loop.
    static let pipelinedDecodeEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_STT_PIPELINED_DECODE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw != "0" && raw != "false" && raw != "off"
    }()

    public init(modelId: String = Qwen3ASRResources.defaultModelId) {
        self.modelId = modelId
    }

    // MARK: - Public API

    /// Transcribe audio to text
    public func transcribe(
        _ request: ASRRequest,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        try await transcribe(request, modelPath: nil, progressHandler: progressHandler)
    }

    /// Transcribe with explicit model path
    public func transcribe(
        _ request: ASRRequest,
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)

        if loadedModelPath != rootURL.path {
            progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading Qwen3-ASR model..."))
            try await loadModels(from: rootURL, progressHandler: progressHandler)
        }

        progressHandler?(ASRProgress(stage: .loadingAudio, message: "Loading audio..."))
        let audio = try AudioReader.readAudio(from: request.audioURL)
        let streamingRequest = ASRStreamingRequest(
            language: request.language,
            task: request.task,
            maxTokens: request.maxTokens,
            sampleRate: Qwen3ASRResources.sampleRate
        )
        return try await decodeSamples(
            audio,
            request: streamingRequest,
            progressHandler: progressHandler
        )
    }

    public func makeStreamingSession(
        _ request: ASRStreamingRequest
    ) async throws -> any ASRStreamingSession {
        guard request.sampleRate == Qwen3ASRResources.sampleRate else {
            throw ASRStreamingError.invalidInput(
                "Qwen3 ASR streaming expects sampleRate=\(Qwen3ASRResources.sampleRate)."
            )
        }
        guard request.decodeIntervalMs > 0 else {
            throw ASRStreamingError.invalidInput("decodeIntervalMs must be > 0.")
        }
        guard request.minDecodeAudioMs >= 0 else {
            throw ASRStreamingError.invalidInput("minDecodeAudioMs must be >= 0.")
        }
        guard request.maxQueuedAudioMs > 0 else {
            throw ASRStreamingError.invalidInput("maxQueuedAudioMs must be > 0.")
        }

        if thinker == nil || tokenizer == nil || melExtractor == nil || modelConfig == nil {
            try await prepare()
        }
        return Qwen3ASRStreamingSession(
            request: request,
            decode: { [generator = self] samples, melSpec in
                try await generator.decodeSamplesDetailed(
                    samples,
                    request: request,
                    precomputedMelSpec: melSpec,
                    progressHandler: nil
                )
            }
        )
    }

    /// Pre-load models without transcribing
    public func prepare(
        modelPath: String? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws {
        let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
        if loadedModelPath != rootURL.path {
            progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading Qwen3-ASR model..."))
            try await loadModels(from: rootURL, progressHandler: progressHandler)
        }
    }

    /// Unload models from memory
    public func unload() {
        thinker = nil
        tokenizer = nil
        melExtractor = nil
        modelConfig = nil
        loadedModelPath = nil
        Memory.clearCache()
    }
}
