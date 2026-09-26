import AudioBreezeTTSModel
import AudioCore
import Foundation
import MLX
import MLXNN
import MereRunCore

/// Owns one resident Breeze checkpoint. Model evaluation is serialized by the actor.
public actor BreezeTTSGenerator {
    private let modelID: String
    private var model: BreezeTTSModel?
    private var loadedPath: String?
    private let streams = Stream.Context()

    public init(modelID: String = ManagedModelID.breezeTTS2.rawValue) {
        self.modelID = modelID
    }

    public func generateAudio(
        _ request: TTSRequest,
        modelPath: String?,
        progressHandler: (@Sendable (TTSProgress) -> Void)?
    ) async throws -> AudioWaveform {
        try await synthesizeAudio(
            request, modelPath: modelPath, progressHandler: progressHandler,
            streamingOptions: nil, continuation: nil
        )
    }

    nonisolated public func generateStream(
        _ request: TTSRequest,
        options: TTSStreamingOptions,
        modelPath: String?
    ) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { [self] in
                do {
                    let audio = try await synthesizeAudio(
                        request, modelPath: modelPath, progressHandler: nil,
                        streamingOptions: options, continuation: continuation
                    )
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
        _ request: TTSRequest,
        modelPath: String?,
        progressHandler: (@Sendable (TTSProgress) -> Void)?,
        streamingOptions: TTSStreamingOptions?,
        continuation: AsyncThrowingStream<TTSStreamingEvent, Error>.Continuation?
    ) async throws -> AudioWaveform {
        try await Stream.withDefaultStream(streams) {
            defer {
                streams.synchronize()
                Memory.clearCache()
            }
            guard request.speaker == nil else {
                throw BreezeTTSError.invalidRequest("Breeze TTS 2 does not provide named speakers.")
            }
            guard request.voiceMode != .clone || request.cloneReference != nil else {
                throw BreezeTTSError.invalidRequest("Voice cloning requires reference audio and its exact transcript.")
            }
            let resolved = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: modelPath ?? modelID,
                defaultModelID: modelID
            )
            if loadedPath != resolved.url.path {
                progressHandler?(TTSProgress(stage: .loadingModel, message: "Loading Breeze TTS 2 model..."))
                model = try await BreezeTTSModel.fromModelDirectory(resolved.url)
                loadedPath = resolved.url.path
            }
            guard let model else { throw BreezeTTSError.missingAsset("model") }
            let reference: MLXArray?
            let transcript: String?
            if request.voiceMode == .clone {
                guard let clone = request.cloneReference else {
                    throw BreezeTTSError.invalidRequest("Voice cloning requires reference audio and its exact transcript.")
                }
                progressHandler?(TTSProgress(stage: .preprocessingReference, message: "Preparing reference audio..."))
                let processed = try Qwen3TTSAudioPreprocessor.loadAndProcess(from: clone.audioURL)
                reference = MLXArray(processed.samples)
                transcript = clone.transcript
            } else {
                reference = nil
                transcript = nil
            }
            progressHandler?(TTSProgress(stage: .generating, message: "Generating Breeze speech..."))
            var frameCount = 0
            let audio = try model.generate(
                text: request.text,
                instruction: request.voiceDescription,
                referenceAudio: reference,
                referenceTranscript: transcript,
                parameters: BreezeGenerationParameters(temperature: request.temperature),
                onToken: { token in
                    frameCount += 1
                    if streamingOptions?.emitTokenEvents == true {
                        continuation?.yield(.token(id: token))
                    }
                },
                chunkFrameInterval: streamingOptions?.chunkTokenInterval,
                onAudioChunk: streamingOptions == nil ? nil : { samples in
                    continuation?.yield(.audioChunk(samples: samples, sampleRate: model.sampleRate))
                }
            )
            try Task.checkCancellation()
            progressHandler?(TTSProgress(stage: .decoding, tokensGenerated: frameCount))
            eval(audio)
            return try AudioWaveform(
                interleaved: audio.reshaped(-1).asArray(Float.self), channels: 1, sampleRate: model.sampleRate
            )
        }
    }

    public func unload() {
        streams.synchronize()
        model = nil
        loadedPath = nil
        Memory.clearCache()
    }
}

public struct BreezeTTSSynthesisExecutor: SpeechSynthesisExecutor {
    private let generator: BreezeTTSGenerator
    private let modelPath: String?

    public init(generator: BreezeTTSGenerator, modelPath: String?) {
        self.generator = generator
        self.modelPath = modelPath
    }

    public func generate(
        _ request: TTSRequest,
        progressHandler: (@Sendable (TTSProgress) -> Void)?
    ) async throws -> AudioWaveform {
        try await generator.generateAudio(request, modelPath: modelPath, progressHandler: progressHandler)
    }

    public func generateStream(
        _ request: TTSRequest,
        options: TTSStreamingOptions
    ) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        generator.generateStream(request, options: options, modelPath: modelPath)
    }
}
