import AudioCore
import AudioCodecs
import AudioDrama3Model
import Foundation
import MLX
import MereRunCore
@preconcurrency import Tokenizers

#if !os(iOS)
public actor Drama3Generator {
    private var model: Drama3Model?
    private var codec: Drama3Codec?
    private var tokenizer: (any Tokenizers.Tokenizer)?
    private let streams = Stream.Context()

    public init() {}

    public func generateAudio(_ request: TTSRequest, modelPath: String?,
                              progressHandler: (@Sendable (TTSProgress) -> Void)?) async throws -> AudioWaveform {
        try await synthesize(request, modelPath: modelPath, progressHandler: progressHandler, options: nil, continuation: nil)
    }

    nonisolated public func generateStream(_ request: TTSRequest, options: TTSStreamingOptions,
                                          modelPath: String?) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let audio = try await self.synthesize(request, modelPath: modelPath, progressHandler: nil,
                                                         options: options, continuation: continuation)
                    try Task.checkCancellation()
                    continuation.yield(.completed(result: TTSResult(audioURL: request.outputURL,
                        duration: audio.duration, sampleRate: audio.sampleRate)))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable reason in if case .cancelled = reason { task.cancel() } }
        }
    }

    private func synthesize(_ request: TTSRequest, modelPath: String?,
                            progressHandler: (@Sendable (TTSProgress) -> Void)?, options: TTSStreamingOptions?,
                            continuation: AsyncThrowingStream<TTSStreamingEvent, Error>.Continuation?) async throws -> AudioWaveform {
        try Drama3Prompt.validate(request.text)
        guard request.speaker == nil, request.cfgScale == nil else {
            throw Drama3Error.invalid("use inline <|speaker:N|> and event tags; named speakers and CFG are unsupported")
        }
        guard request.voiceMode != .clone || request.cloneReference != nil else {
            throw Drama3Error.invalid("voice cloning requires reference audio and its exact transcript")
        }
        return try await Stream.withDefaultStream(streams) {
            defer { unload() }
            let resolved = try await ManagedModelResolver.resolveForRuntime(requestedModel: modelPath ?? Drama3Resources.modelID,
                defaultModelID: Drama3Resources.modelID)
            var references: [Drama3Prompt.Reference] = []
            if request.voiceMode == .clone, let reference = request.cloneReference {
                try Drama3Prompt.validate(reference.transcript)
                progressHandler?(TTSProgress(stage: .preprocessingReference, message: "Encoding reference speech..."))
                let audio = try AudioReader.readAudioBuffer(from: reference.audioURL, sampleRate: Drama3Codec.sampleRate, channels: 1)
                guard !audio.samples.isEmpty, audio.samples.count <= Drama3Codec.sampleRate * 30 else {
                    throw Drama3Error.invalid("reference audio must be nonempty and at most 30 seconds")
                }
                codec = try Drama3Resources.loadCodec(root: resolved.url)
                references = [.init(transcript: reference.transcript, codes: try codec!.encode(audio.samples))]
            }
            // Streaming needs both graphs. Batch generation stores integer codes
            // between phases so neither reference encoding nor final decoding
            // overlaps the much larger text/tower checkpoint.
            if options == nil { codec = nil }
            if options != nil, codec == nil { codec = try Drama3Resources.loadCodec(root: resolved.url) }
            Memory.clearCache()
            progressHandler?(TTSProgress(stage: .loadingModel, message: "Loading native DRAMA-3 Q4 experts..."))
            try await loadModel(root: resolved.url)
            Memory.clearCache()
            progressHandler?(TTSProgress(stage: .generating, message: "Generating DRAMA-3 speech..."))
            let generated = try generateCodes(request, references: references, options: options,
                continuation: continuation, progressHandler: progressHandler)
            if options == nil {
                // No local strong model/tokenizer binding may survive this phase.
                model = nil; tokenizer = nil
                streams.synchronize()
                Memory.clearCache()
            }
            let codes = generated.codes
            progressHandler?(TTSProgress(stage: .decoding, tokensGenerated: codes[0].count,
                message: "Decoding DRAMA-3 speech..."))
            if codec == nil { codec = try Drama3Resources.loadCodec(root: resolved.url) }
            let decoded = try codec!.decode(codes)
            MLX.eval(decoded)
            let samples = decoded.asArray(Float.self)
            if options != nil, samples.count > generated.emittedSamples {
                continuation?.yield(.audioChunk(samples: Array(samples.dropFirst(generated.emittedSamples)),
                    sampleRate: Drama3Codec.sampleRate))
            }
            return try AudioWaveform(interleaved: samples, channels: 1, sampleRate: Drama3Codec.sampleRate)
        }
    }

    private func loadModel(root: URL) async throws {
        let loaded = try Drama3Model(config: Drama3Resources.configuration(root: root))
        try loaded.load(root: root)
        model = loaded
        tokenizer = try await AutoTokenizer.from(modelFolder: root)
    }

    private func generateCodes(_ request: TTSRequest, references: [Drama3Prompt.Reference],
                               options: TTSStreamingOptions?,
                               continuation: AsyncThrowingStream<TTSStreamingEvent, Error>.Continuation?,
                               progressHandler: (@Sendable (TTSProgress) -> Void)?) throws -> (codes: [[Int32]], emittedSamples: Int) {
        guard let model, let tokenizer else { throw Drama3Error.invalid("checkpoint did not load") }
        let prompt = try Drama3Prompt.rows(text: request.text, references: references, config: model.config,
            encode: { tokenizer.encode(text: $0, addSpecialTokens: false) })
        var frames: [[Int32]] = [], emittedSamples = 0
        let codes = try model.generate(prompt: prompt, temperature: request.temperature, seed: request.seed, onFrame: { frame in
            frames.append(frame)
            if frames.count.isMultiple(of: 25) {
                progressHandler?(TTSProgress(stage: .generating, tokensGenerated: frames.count,
                    message: "Generating DRAMA-3 speech..."))
            }
            if options?.emitTokenEvents == true { continuation?.yield(.token(id: Int(frame[0]))) }
            if let options, let codec = self.codec, frames.count.isMultiple(of: options.chunkTokenInterval) {
                let partial = try codec.decode((0..<10).map { b in frames.map { $0[b] } })
                MLX.eval(partial)
                let values = partial.asArray(Float.self)
                continuation?.yield(.audioChunk(samples: Array(values.dropFirst(emittedSamples)),
                    sampleRate: Drama3Codec.sampleRate))
                emittedSamples = values.count
            }
        })
        return (codes, emittedSamples)
    }

    public func unload() {
        streams.synchronize()
        model = nil; codec = nil; tokenizer = nil
        Memory.clearCache()
    }
}

#else
public actor Drama3Generator {
    public init() {}
    public func generateAudio(_ request: TTSRequest, modelPath: String?,
                              progressHandler: (@Sendable (TTSProgress) -> Void)?) throws -> AudioWaveform {
        throw Drama3Error.invalid("native inference requires macOS or Linux")
    }
    nonisolated public func generateStream(_ request: TTSRequest, options: TTSStreamingOptions,
                                          modelPath: String?) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        AsyncThrowingStream { $0.finish(throwing: Drama3Error.invalid("native inference requires macOS or Linux")) }
    }
    public func unload() {}
}
#endif

public struct Drama3SynthesisExecutor: SpeechSynthesisExecutor {
    private let generator: Drama3Generator
    private let modelPath: String?
    public init(generator: Drama3Generator, modelPath: String?) { self.generator = generator; self.modelPath = modelPath }
    public func generate(_ request: TTSRequest, progressHandler: (@Sendable (TTSProgress) -> Void)?) async throws -> AudioWaveform {
        try await generator.generateAudio(request, modelPath: modelPath, progressHandler: progressHandler)
    }
    public func generateStream(_ request: TTSRequest, options: TTSStreamingOptions) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        generator.generateStream(request, options: options, modelPath: modelPath)
    }
}
