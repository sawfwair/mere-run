import Foundation

// MARK: - TTS Types

public enum TTSVoiceMode: String, Sendable, Hashable, Codable {
    case style
    case clone
}

public struct TTSCloneReference: Sendable, Hashable, Codable {
    public var audioURL: URL
    public var transcript: String
    public var language: String?
    public var profileID: UUID?

    public init(
        audioURL: URL,
        transcript: String,
        language: String? = nil,
        profileID: UUID? = nil
    ) {
        self.audioURL = audioURL
        self.transcript = transcript
        self.language = language
        self.profileID = profileID
    }
}

public struct TTSRequest: Sendable, Hashable {
    public var text: String
    public var voiceDescription: String
    public var voiceMode: TTSVoiceMode
    public var cloneReference: TTSCloneReference?
    public var language: String
    public var speed: Float
    public var temperature: Float
    public var outputURL: URL

    public init(
        text: String,
        voiceDescription: String = "A calm female voice with clear pronunciation",
        voiceMode: TTSVoiceMode = .style,
        cloneReference: TTSCloneReference? = nil,
        language: String = "auto",
        speed: Float = 1.0,
        temperature: Float = 0.6,
        outputURL: URL
    ) {
        self.text = text
        self.voiceDescription = voiceDescription
        self.voiceMode = voiceMode
        self.cloneReference = cloneReference
        self.language = language
        self.speed = speed
        self.temperature = temperature
        self.outputURL = outputURL
    }
}

public struct TTSResult: Sendable, Hashable, Codable {
    public let audioURL: URL
    public let duration: TimeInterval
    public let sampleRate: Int

    public init(audioURL: URL, duration: TimeInterval, sampleRate: Int = 24000) {
        self.audioURL = audioURL
        self.duration = duration
        self.sampleRate = sampleRate
    }
}

public enum TTSStage: String, Sendable, Hashable {
    case loadingModel
    case preprocessingReference
    case encodingReference
    case buildingPrompt
    case tokenizing
    case generating
    case decoding
    case saving
}

public struct TTSProgress: Sendable, Hashable {
    public let stage: TTSStage
    public let tokensGenerated: Int
    public let message: String?

    public init(stage: TTSStage, tokensGenerated: Int = 0, message: String? = nil) {
        self.stage = stage
        self.tokensGenerated = tokensGenerated
        self.message = message
    }
}

public struct TTSStreamingOptions: Sendable, Hashable, Codable {
    public var chunkTokenInterval: Int
    public var emitTokenEvents: Bool

    public init(
        chunkTokenInterval: Int = 25,
        emitTokenEvents: Bool = true
    ) {
        self.chunkTokenInterval = chunkTokenInterval
        self.emitTokenEvents = emitTokenEvents
    }
}

public enum TTSStreamingEvent: Sendable, Hashable, Codable {
    case token(id: Int)
    case audioChunk(samples: [Float], sampleRate: Int)
    case completed(result: TTSResult)
}

public protocol TTSGenerator {
    func generate(
        _ request: TTSRequest,
        progressHandler: (@Sendable (TTSProgress) -> Void)?
    ) async throws -> TTSResult

    func generateStream(
        _ request: TTSRequest,
        options: TTSStreamingOptions
    ) -> AsyncThrowingStream<TTSStreamingEvent, Error>
}

public extension TTSGenerator {
    func generateStream(
        _ request: TTSRequest,
        options: TTSStreamingOptions = TTSStreamingOptions()
    ) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(
                throwing: ASRStreamingError.unsupportedBackend(
                    "\(Self.self) does not support streaming generation."
                )
            )
        }
    }
}

// MARK: - ASR Types

public struct ASRRequest: Sendable, Hashable {
    public var audioURL: URL
    public var language: String?
    public var task: ASRTask
    public var maxTokens: Int

    public init(
        audioURL: URL,
        language: String? = nil,
        task: ASRTask = .transcribe,
        maxTokens: Int = 448
    ) {
        self.audioURL = audioURL
        self.language = language
        self.task = task
        self.maxTokens = maxTokens
    }
}

public enum ASRTask: String, Sendable, Hashable, Codable {
    case transcribe
    case translate
}

public struct ASRTokenAlignment: Sendable, Hashable, Codable {
    public let id: Int?
    public let text: String
    public let startSeconds: TimeInterval
    public let durationSeconds: TimeInterval
    public let endSeconds: TimeInterval

    public init(
        id: Int? = nil,
        text: String,
        startSeconds: TimeInterval,
        durationSeconds: TimeInterval,
        endSeconds: TimeInterval? = nil
    ) {
        self.id = id
        self.text = text
        self.startSeconds = startSeconds
        self.durationSeconds = durationSeconds
        self.endSeconds = endSeconds ?? (startSeconds + durationSeconds)
    }

    enum CodingKeys: String, CodingKey {
        case id
        case text
        case startSeconds = "start_seconds"
        case durationSeconds = "duration_seconds"
        case endSeconds = "end_seconds"
    }
}

public struct ASRSentenceAlignment: Sendable, Hashable, Codable {
    public let text: String
    public let startSeconds: TimeInterval
    public let durationSeconds: TimeInterval
    public let endSeconds: TimeInterval
    public let tokens: [ASRTokenAlignment]

    public init(
        text: String,
        startSeconds: TimeInterval,
        durationSeconds: TimeInterval,
        endSeconds: TimeInterval? = nil,
        tokens: [ASRTokenAlignment]
    ) {
        self.text = text
        self.startSeconds = startSeconds
        self.durationSeconds = durationSeconds
        self.endSeconds = endSeconds ?? (startSeconds + durationSeconds)
        self.tokens = tokens
    }

    enum CodingKeys: String, CodingKey {
        case text
        case startSeconds = "start_seconds"
        case durationSeconds = "duration_seconds"
        case endSeconds = "end_seconds"
        case tokens
    }
}

public struct ASRResult: Sendable, Hashable, Codable {
    public let text: String
    public let language: String?
    public let duration: TimeInterval
    public let tokenAlignments: [ASRTokenAlignment]?
    public let sentenceAlignments: [ASRSentenceAlignment]?

    public init(
        text: String,
        language: String? = nil,
        duration: TimeInterval = 0,
        tokenAlignments: [ASRTokenAlignment]? = nil,
        sentenceAlignments: [ASRSentenceAlignment]? = nil
    ) {
        self.text = text
        self.language = language
        self.duration = duration
        self.tokenAlignments = tokenAlignments
        self.sentenceAlignments = sentenceAlignments
    }

    enum CodingKeys: String, CodingKey {
        case text
        case language
        case duration
        case tokenAlignments = "token_alignments"
        case sentenceAlignments = "sentence_alignments"
    }
}

public enum ASRStage: String, Sendable, Hashable {
    case loadingModel
    case loadingAudio
    case extractingFeatures
    case transcribing
}

public struct ASRProgress: Sendable, Hashable {
    public let stage: ASRStage
    public let tokensGenerated: Int
    public let message: String?

    public init(stage: ASRStage, tokensGenerated: Int = 0, message: String? = nil) {
        self.stage = stage
        self.tokensGenerated = tokensGenerated
        self.message = message
    }
}

public struct ASRStreamingRequest: Sendable, Hashable, Codable {
    public var language: String?
    public var task: ASRTask
    public var maxTokens: Int
    public var sampleRate: Int
    public var decodeIntervalMs: Int
    public var minDecodeAudioMs: Int
    public var maxQueuedAudioMs: Int

    public init(
        language: String? = nil,
        task: ASRTask = .transcribe,
        maxTokens: Int = 448,
        sampleRate: Int = 16_000,
        decodeIntervalMs: Int = 500,
        minDecodeAudioMs: Int = 800,
        maxQueuedAudioMs: Int = 5_000
    ) {
        self.language = language
        self.task = task
        self.maxTokens = maxTokens
        self.sampleRate = sampleRate
        self.decodeIntervalMs = decodeIntervalMs
        self.minDecodeAudioMs = minDecodeAudioMs
        self.maxQueuedAudioMs = maxQueuedAudioMs
    }
}

public struct ASRStreamingStats: Sendable, Hashable, Codable {
    public let decodeCount: Int
    public let totalAudioSeconds: Double
    public let lastDecodeLatencyMs: Double
    public let tokensGenerated: Int

    public init(
        decodeCount: Int,
        totalAudioSeconds: Double,
        lastDecodeLatencyMs: Double,
        tokensGenerated: Int
    ) {
        self.decodeCount = decodeCount
        self.totalAudioSeconds = totalAudioSeconds
        self.lastDecodeLatencyMs = lastDecodeLatencyMs
        self.tokensGenerated = tokensGenerated
    }
}

public enum ASRStreamingEvent: Sendable, Hashable, Codable {
    case partial(text: String)
    case final(result: ASRResult)
    case stats(ASRStreamingStats)
}

public protocol ASRStreamingSession: Sendable {
    var events: AsyncThrowingStream<ASRStreamingEvent, Error> { get }

    func feed(samples: [Float]) async throws
    func finish() async throws
    /// Finish once a decode covers the required prefix of the session audio.
    /// Implementations that cannot distinguish trailing audio may use `finish()`.
    func finish(requiredSampleCount: Int) async throws
    func cancel() async
}

public extension ASRStreamingSession {
    func finish(requiredSampleCount _: Int) async throws {
        try await finish()
    }
}

public enum ASRStreamingError: Error, LocalizedError, Sendable, Hashable {
    case unsupportedBackend(String)
    case invalidInput(String)
    case invalidState(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedBackend(let message):
            return message
        case .invalidInput(let message):
            return message
        case .invalidState(let message):
            return message
        }
    }
}

public protocol ASRGenerator {
    func transcribe(
        _ request: ASRRequest,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult

    func makeStreamingSession(
        _ request: ASRStreamingRequest
    ) async throws -> any ASRStreamingSession
}

public extension ASRGenerator {
    func makeStreamingSession(
        _ request: ASRStreamingRequest
    ) async throws -> any ASRStreamingSession {
        throw ASRStreamingError.unsupportedBackend(
            "\(Self.self) does not support streaming transcription."
        )
    }
}
