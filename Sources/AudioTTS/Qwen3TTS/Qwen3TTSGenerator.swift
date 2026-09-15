import AudioQwen3TTSModel
import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioCore
import AudioCodecs
import MereRunCore

/// Owns Qwen model state and the shared native style/clone waveform path.
public actor Qwen3TTSGenerator: TTSGenerator {
    var talker: Qwen3TTSTalkerForConditionalGeneration?
    var speechTokenizer: Qwen3TTSSpeechTokenizer?
    var speakerEncoder: Qwen3TTSSpeakerEncoder?
    var tokenizer: Qwen3TTSTokenizer?
    var modelConfig: Qwen3TTSModelConfig?
    var loadedModelPath: String?
    let modelId: String
    private let streams = Stream.Context()

    public init(modelId: String = Qwen3TTSResources.defaultModelId) {
        self.modelId = modelId
    }

    public func generate(
        _ request: TTSRequest,
        progressHandler: (@Sendable (TTSProgress) -> Void)? = nil
    ) async throws -> TTSResult {
        try await generate(request, modelPath: nil, progressHandler: progressHandler)
    }

    /// Preserves the public file-generation entry point through shared PCM16 export.
    public func generate(
        _ request: TTSRequest,
        modelPath: String?,
        progressHandler: (@Sendable (TTSProgress) -> Void)? = nil
    ) async throws -> TTSResult {
        let plan = try SpeechSynthesisPlan(request: request)
        return try await SpeechSynthesisOperation.execute(
            plan, executor: Qwen3TTSSynthesisExecutor(generator: self, modelPath: modelPath),
            progressHandler: progressHandler
        ).result
    }

    /// Returns evaluated host samples. The shared operation owns file publication.
    public func generateAudio(
        _ request: TTSRequest,
        modelPath: String? = nil,
        progressHandler: (@Sendable (TTSProgress) -> Void)? = nil
    ) async throws -> AudioWaveform {
        let plan = try SpeechSynthesisPlan(request: request)
        return try await synthesizeAudio(plan, modelPath: modelPath, progressHandler: progressHandler, continuation: nil)
    }

    nonisolated public func generateStream(
        _ request: TTSRequest,
        options: TTSStreamingOptions
    ) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        generateStream(request, options: options, modelPath: nil)
    }

    /// Preserves the public sample-event stream. Use SpeechSynthesisOperation to publish a WAV.
    nonisolated public func generateStream(
        _ request: TTSRequest,
        options: TTSStreamingOptions,
        modelPath: String? = nil
    ) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { [self] in
                do {
                    let plan = try SpeechSynthesisPlan(request: request, streamingOptions: options)
                    let audio = try await synthesizeAudio(plan, modelPath: modelPath, progressHandler: nil, continuation: continuation)
                    try Task.checkCancellation()
                    continuation.yield(.completed(result: TTSResult(
                        audioURL: request.outputURL, duration: audio.duration, sampleRate: audio.sampleRate
                    )))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable reason in
                if case .cancelled = reason { task.cancel() }
            }
        }
    }

    private func synthesizeAudio(
        _ plan: SpeechSynthesisPlan,
        modelPath: String?,
        progressHandler: (@Sendable (TTSProgress) -> Void)?,
        continuation: AsyncThrowingStream<TTSStreamingEvent, Error>.Continuation?
    ) async throws -> AudioWaveform {
        try plan.validateForExecution()
        return try await Stream.withDefaultStream(streams) {
            defer {
                streams.synchronize()
                Memory.clearCache()
            }
            let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
            try Task.checkCancellation()
            if loadedModelPath != rootURL.path {
                progressHandler?(TTSProgress(stage: .loadingModel, message: "Loading Qwen3-TTS model..."))
                try await loadModels(from: rootURL, progressHandler: progressHandler)
            }
            try Task.checkCancellation()
            return try generateLoadedAudio(plan, rootURL: rootURL, progressHandler: progressHandler, continuation: continuation)
        }
    }

    private func generateLoadedAudio(
        _ plan: SpeechSynthesisPlan,
        rootURL: URL,
        progressHandler: (@Sendable (TTSProgress) -> Void)?,
        continuation: AsyncThrowingStream<TTSStreamingEvent, Error>.Continuation?
    ) throws -> AudioWaveform {
        guard let talker, let speechTokenizer, let tokenizer, let modelConfig else {
            throw Qwen3TTSError.modelsNotLoaded
        }
        let request = plan.request
        if request.voiceMode == .clone {
            let missing = Qwen3TTSResources(rootURL: rootURL).validateCloneAssets()
            if !missing.isEmpty && speakerEncoder == nil {
                throw Qwen3TTSError.cloneAssetsMissing(missing.map { $0.lastPathComponent })
            }
        }
        let onToken: ((Int) -> Void)? = plan.streamingOptions?.emitTokenEvents == true
            ? { token in continuation?.yield(.token(id: token)) }
            : nil
        let onAudioDelta: (([Float]) -> Void)? = continuation.map { continuation in
            { samples in
                if !samples.isEmpty {
                    continuation.yield(.audioChunk(samples: samples, sampleRate: modelConfig.sampleRate))
                }
            }
        }
        let audio: MLXArray
        switch request.voiceMode {
        case .style:
            progressHandler?(TTSProgress(stage: .tokenizing, message: "Preparing inputs..."))
            audio = try generateVoiceDesign(
                text: request.text, language: request.language, instruct: request.voiceDescription,
                speakerHintTokens: nil, referencePromptTokens: nil,
                talker: talker, tokenizer: tokenizer, speechTokenizer: speechTokenizer, config: modelConfig,
                temperature: request.temperature, progressHandler: progressHandler,
                streamingChunkTokenInterval: plan.streamingOptions?.chunkTokenInterval,
                onToken: onToken, onAudioDelta: onAudioDelta
            )
        case .clone:
            audio = try generateVoiceClone(
                request: request, talker: talker, tokenizer: tokenizer, speechTokenizer: speechTokenizer,
                speakerEncoder: speakerEncoder, config: modelConfig, progressHandler: progressHandler,
                streamingChunkTokenInterval: plan.streamingOptions?.chunkTokenInterval,
                onToken: onToken, onAudioDelta: onAudioDelta
            )
        }
        try Task.checkCancellation()
        MLX.eval(audio)
        return try AudioWaveform(interleaved: audio.reshaped(-1).asArray(Float.self), channels: 1, sampleRate: modelConfig.sampleRate)
    }

    public func prepare(
        modelPath: String? = nil,
        progressHandler: (@Sendable (TTSProgress) -> Void)? = nil
    ) async throws {
        try Task.checkCancellation()
        try await Stream.withDefaultStream(streams) {
            defer { streams.synchronize() }
            let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
            try Task.checkCancellation()
            if loadedModelPath != rootURL.path {
                progressHandler?(TTSProgress(stage: .loadingModel, message: "Loading Qwen3-TTS model..."))
                try await loadModels(from: rootURL, progressHandler: progressHandler)
            }
        }
    }

    public func unload() {
        streams.synchronize()
        talker = nil
        speechTokenizer = nil
        speakerEncoder = nil
        tokenizer = nil
        modelConfig = nil
        loadedModelPath = nil
        Memory.clearCache()
    }
}
