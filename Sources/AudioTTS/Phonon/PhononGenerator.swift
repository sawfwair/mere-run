import AudioCore
import AudioPhononModel
import Foundation
import MLX
import MereRunCore

public actor PhononGenerator {
    private let modelID: String
    private var model: PhononModel?
    private var loadedPath: String?
    private let streams = Stream.Context()

    public init(modelID: String = ManagedModelID.phonon31m.rawValue) { self.modelID = modelID }

    /// Validate before resolving or downloading a checkpoint.
    public nonisolated static func validate(_ request: TTSRequest) throws {
        try SpeechSynthesisPlan.validateParameters(text: request.text, temperature: request.temperature, speed: request.speed)
        guard request.voiceMode == .style, request.cloneReference == nil else {
            throw PhononError.invalidRequest("Phonon accepts precomputed voices; raw-audio voice cloning is not supported.")
        }
        guard request.voiceDescription.isEmpty || request.voiceDescription == TTSRequest.defaultVoiceDescription,
              request.speaker == nil || request.speaker == "default" else {
            throw PhononError.invalidRequest("Phonon uses the checkpoint's default voice. Text voice descriptions and named speakers are not supported.")
        }
        guard ["auto", "en", "english", "en-us", "en-gb"].contains(request.language.lowercased()),
              request.speed == 1, request.cfgScale == nil else {
            throw PhononError.invalidRequest("Phonon 31m supports English at its native speaking rate without CFG guidance.")
        }
    }

    public func generateAudio(_ request: TTSRequest, modelPath: String?,
                              progressHandler: (@Sendable (TTSProgress) -> Void)?) async throws -> AudioWaveform {
        try await synthesize(request, modelPath: modelPath, progressHandler: progressHandler, onFrame: nil)
    }

    private func synthesize(_ request: TTSRequest, modelPath: String?,
                            progressHandler: (@Sendable (TTSProgress) -> Void)?,
                            onFrame: (([Float], Int) -> Void)?) async throws -> AudioWaveform {
        try Self.validate(request)
        return try await Stream.withDefaultStream(streams) {
            defer { streams.synchronize(); Memory.clearCache() }
            let resolved = try await ManagedModelResolver.resolveForRuntime(requestedModel: modelPath ?? modelID, defaultModelID: modelID)
            if loadedPath != resolved.url.path {
                progressHandler?(TTSProgress(stage: .loadingModel, message: "Loading Phonon 31m..."))
                model = try PhononModel(directory: resolved.url)
                loadedPath = resolved.url.path
            }
            guard let model else { throw PhononError.invalidAsset("model is not loaded") }
            let voice = try model.loadVoice(url: resolved.url.appendingPathComponent("default-voice.safetensors"))
            progressHandler?(TTSProgress(stage: .generating, message: "Generating Phonon speech..."))
            var samples: [Float] = []
            try model.generate(text: request.text, voice: voice, temperature: request.temperature, seed: request.seed) { frame in
                samples.append(contentsOf: frame)
                onFrame?(frame, model.sampleRate)
            }
            return try AudioWaveform(interleaved: samples, channels: 1, sampleRate: model.sampleRate)
        }
    }

    nonisolated public func generateStream(_ request: TTSRequest, options: TTSStreamingOptions,
                                           modelPath: String?) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { [self] in
                do {
                    try SpeechSynthesisPlan.validateStreamingOptions(options)
                    let audio = try await synthesize(request, modelPath: modelPath, progressHandler: nil) { samples, rate in
                        continuation.yield(.audioChunk(samples: samples, sampleRate: rate))
                    }
                    try Task.checkCancellation()
                    continuation.yield(.completed(result: TTSResult(audioURL: request.outputURL, duration: audio.duration, sampleRate: audio.sampleRate)))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable reason in if case .cancelled = reason { task.cancel() } }
        }
    }
    public func unload() {
        streams.synchronize(); model = nil; loadedPath = nil; Memory.clearCache()
    }
}

public struct PhononSynthesisExecutor: SpeechSynthesisExecutor {
    let generator: PhononGenerator
    let modelPath: String?
    public func generate(_ request: TTSRequest, progressHandler: (@Sendable (TTSProgress) -> Void)?) async throws -> AudioWaveform {
        try await generator.generateAudio(request, modelPath: modelPath, progressHandler: progressHandler)
    }
    public func generateStream(_ request: TTSRequest, options: TTSStreamingOptions) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        generator.generateStream(request, options: options, modelPath: modelPath)
    }
}
