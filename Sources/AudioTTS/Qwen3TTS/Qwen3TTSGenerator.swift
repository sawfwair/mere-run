import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioCore
import AudioCodecs
import MereRunCore

/// Owns the public Qwen3 TTS entrypoints and loaded runtime state.
/// Model loading and generation internals live in companion files so the
/// speech flow reads from entrypoint to implementation in a predictable order.
public actor Qwen3TTSGenerator: TTSGenerator {
    var talker: Qwen3TTSTalkerForConditionalGeneration?
    var speechTokenizer: Qwen3TTSSpeechTokenizer?
    var speakerEncoder: Qwen3TTSSpeakerEncoder?
    var tokenizer: Qwen3TTSTokenizer?
    var modelConfig: Qwen3TTSModelConfig?
    var loadedModelPath: String?
    let modelId: String

    public init(modelId: String = Qwen3TTSResources.defaultModelId) {
        self.modelId = modelId
    }

    public func generate(
        _ request: TTSRequest,
        progressHandler: (@Sendable (TTSProgress) -> Void)? = nil
    ) async throws -> TTSResult {
        try await generate(request, modelPath: nil, progressHandler: progressHandler)
    }

    public func generate(
        _ request: TTSRequest,
        modelPath: String?,
        progressHandler: (@Sendable (TTSProgress) -> Void)? = nil
    ) async throws -> TTSResult {
        let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)

        if loadedModelPath != rootURL.path {
            progressHandler?(TTSProgress(stage: .loadingModel, message: "Loading Qwen3-TTS model..."))
            try await loadModels(from: rootURL, progressHandler: progressHandler)
        }

        guard let talker, let speechTokenizer, let tokenizer, let modelConfig else {
            throw Qwen3TTSError.modelsNotLoaded
        }

        if request.voiceMode == .clone {
            let cloneMissing = Qwen3TTSResources(rootURL: rootURL).validateCloneAssets()
            if !cloneMissing.isEmpty && speakerEncoder == nil {
                throw Qwen3TTSError.cloneAssetsMissing(cloneMissing.map { $0.lastPathComponent })
            }
        }

        let audio: MLXArray
        switch request.voiceMode {
        case .style:
            progressHandler?(TTSProgress(stage: .tokenizing, message: "Preparing inputs..."))
            audio = try generateVoiceDesign(
                text: request.text,
                language: request.language,
                instruct: request.voiceDescription,
                speakerHintTokens: nil,
                referencePromptTokens: nil,
                talker: talker,
                tokenizer: tokenizer,
                speechTokenizer: speechTokenizer,
                config: modelConfig,
                temperature: request.temperature,
                progressHandler: progressHandler
            )
        case .clone:
            audio = try generateVoiceClone(
                request: request,
                talker: talker,
                tokenizer: tokenizer,
                speechTokenizer: speechTokenizer,
                speakerEncoder: speakerEncoder,
                config: modelConfig,
                progressHandler: progressHandler
            )
        }

        progressHandler?(TTSProgress(stage: .saving, message: "Saving audio..."))
        try SNACAudioWriter.writeWAV(audio, to: request.outputURL, sampleRate: modelConfig.sampleRate)

        let duration = TimeInterval(audio.size) / TimeInterval(modelConfig.sampleRate)
        Memory.clearCache()

        return TTSResult(
            audioURL: request.outputURL,
            duration: duration,
            sampleRate: modelConfig.sampleRate
        )
    }

    nonisolated public func generateStream(
        _ request: TTSRequest,
        options: TTSStreamingOptions,
    ) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        generateStream(request, options: options, modelPath: nil)
    }

    nonisolated public func generateStream(
        _ request: TTSRequest,
        options: TTSStreamingOptions,
        modelPath: String? = nil
    ) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [self] in
                do {
                    try await streamGenerate(
                        request: request,
                        options: options,
                        modelPath: modelPath,
                        continuation: continuation
                    )
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    func streamGenerate(
        request: TTSRequest,
        options: TTSStreamingOptions,
        modelPath: String?,
        continuation: AsyncThrowingStream<TTSStreamingEvent, Error>.Continuation
    ) async throws {
        guard options.chunkTokenInterval > 0 else {
            throw ASRStreamingError.invalidInput("chunkTokenInterval must be > 0.")
        }

        let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: nil)
        if loadedModelPath != rootURL.path {
            try await loadModels(from: rootURL, progressHandler: nil)
        }

        guard let talker, let speechTokenizer, let tokenizer, let modelConfig else {
            throw Qwen3TTSError.modelsNotLoaded
        }

        if request.voiceMode == .clone {
            let cloneMissing = Qwen3TTSResources(rootURL: rootURL).validateCloneAssets()
            if !cloneMissing.isEmpty && speakerEncoder == nil {
                throw Qwen3TTSError.cloneAssetsMissing(cloneMissing.map { $0.lastPathComponent })
            }
        }

        let onToken: ((Int) -> Void)? = options.emitTokenEvents
            ? { tokenId in continuation.yield(.token(id: tokenId)) }
            : nil
        let onAudioDelta: ([Float]) -> Void = { samples in
            guard !samples.isEmpty else { return }
            continuation.yield(.audioChunk(samples: samples, sampleRate: modelConfig.sampleRate))
        }

        let audio: MLXArray
        switch request.voiceMode {
        case .style:
            audio = try generateVoiceDesign(
                text: request.text,
                language: request.language,
                instruct: request.voiceDescription,
                speakerHintTokens: nil,
                referencePromptTokens: nil,
                talker: talker,
                tokenizer: tokenizer,
                speechTokenizer: speechTokenizer,
                config: modelConfig,
                temperature: request.temperature,
                progressHandler: nil,
                streamingChunkTokenInterval: options.chunkTokenInterval,
                onToken: onToken,
                onAudioDelta: onAudioDelta
            )
        case .clone:
            audio = try generateVoiceClone(
                request: request,
                talker: talker,
                tokenizer: tokenizer,
                speechTokenizer: speechTokenizer,
                speakerEncoder: speakerEncoder,
                config: modelConfig,
                progressHandler: nil,
                streamingChunkTokenInterval: options.chunkTokenInterval,
                onToken: onToken,
                onAudioDelta: onAudioDelta
            )
        }

        let duration = TimeInterval(audio.size) / TimeInterval(modelConfig.sampleRate)
        Memory.clearCache()

        continuation.yield(
            .completed(
                result: TTSResult(
                    audioURL: request.outputURL,
                    duration: duration,
                    sampleRate: modelConfig.sampleRate
                )
            )
        )
        continuation.finish()
    }

    public func prepare(
        modelPath: String? = nil,
        progressHandler: (@Sendable (TTSProgress) -> Void)? = nil
    ) async throws {
        let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
        if loadedModelPath != rootURL.path {
            progressHandler?(TTSProgress(stage: .loadingModel, message: "Loading Qwen3-TTS model..."))
            try await loadModels(from: rootURL, progressHandler: progressHandler)
        }
    }

    public func unload() {
        talker = nil
        speechTokenizer = nil
        speakerEncoder = nil
        tokenizer = nil
        modelConfig = nil
        loadedModelPath = nil
        Memory.clearCache()
    }
}
