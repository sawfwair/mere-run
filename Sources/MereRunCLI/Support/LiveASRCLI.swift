import Foundation
import AudioCore
import AudioSTT
import MereRunCore

struct CLILiveASRSession {
    let backend: SpeechBackendOption
    let events: AsyncThrowingStream<ASRLiveEvent, Error>
    let feed: @Sendable ([Float]) async throws -> Void
    let finish: @Sendable (ASRLiveFinishReason) async throws -> Void
    let cancel: @Sendable () async -> Void
}

struct PCM16LittleEndianDecoder {
    private var trailingByte: UInt8?

    mutating func decode(_ data: Data) -> [Float] {
        guard !data.isEmpty || trailingByte != nil else { return [] }
        var bytes = [UInt8]()
        bytes.reserveCapacity(data.count + (trailingByte == nil ? 0 : 1))
        if let trailingByte {
            bytes.append(trailingByte)
            self.trailingByte = nil
        }
        bytes.append(contentsOf: data)
        if !bytes.count.isMultiple(of: 2) {
            trailingByte = bytes.removeLast()
        }
        var samples = [Float]()
        samples.reserveCapacity(bytes.count / 2)
        for offset in stride(from: 0, to: bytes.count, by: 2) {
            let word = UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
            samples.append(Float(Int16(bitPattern: word)) / 32_768)
        }
        return samples
    }

    func validateEOF() throws {
        guard trailingByte == nil else {
            throw ASRStreamingError.invalidInput("pcm_odd_byte_eof: PCM stdin ended with an incomplete 16-bit sample.")
        }
    }
}

struct LiveASRProtocolEvent: Encodable {
    let protocolVersion: Int
    let type: String
    var sampleRate: Int?
    var inputFormat: String?
    var utteranceId: String?
    var revision: Int?
    var text: String?
    var startMs: Int?
    var endMs: Int?
    var decodeLatencyMs: Double?
    var audioMs: Int?
    var queuedAudioMs: Int?
    var reason: String?
    var code: String?
    var message: String?

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol"
        case type
        case sampleRate
        case inputFormat
        case utteranceId
        case revision
        case text
        case startMs
        case endMs
        case decodeLatencyMs
        case audioMs
        case queuedAudioMs
        case reason
        case code
        case message
    }

    init(type: String) {
        self.protocolVersion = 1
        self.type = type
    }

    static func ready() -> Self {
        var event = Self(type: "ready")
        event.sampleRate = 16_000
        event.inputFormat = "pcm-s16le"
        return event
    }

    static func transcript(type: String, value: ASRLiveTranscript) -> Self {
        var event = Self(type: type)
        event.utteranceId = value.utteranceId
        event.revision = value.revision
        event.text = value.text
        event.startMs = value.startMs
        event.endMs = value.endMs
        return event
    }

    static func stats(_ value: ASRStreamingStats, queuedAudioMs: Int) -> Self {
        var event = Self(type: "stats")
        event.decodeLatencyMs = value.lastDecodeLatencyMs
        event.audioMs = Int((value.totalAudioSeconds * 1_000).rounded())
        event.queuedAudioMs = queuedAudioMs
        return event
    }

    static func final(_ reason: ASRLiveFinishReason) -> Self {
        var event = Self(type: "final")
        event.reason = reason.rawValue
        return event
    }

    static func error(code: String, message: String) -> Self {
        var event = Self(type: "error")
        event.code = code
        event.message = message
        return event
    }
}

enum LiveASRCLIWriter {
    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }()

    static func write(_ event: LiveASRProtocolEvent) throws {
        let data = try encoder.encode(event)
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([0x0A]))
    }

    static func consume(
        _ events: AsyncThrowingStream<ASRLiveEvent, Error>,
        jsonl: Bool,
        quiet: Bool
    ) async throws {
        for try await event in events {
            switch event {
            case .partial(let transcript):
                if jsonl {
                    try write(.transcript(type: "partial", value: transcript))
                } else if !quiet {
                    CLIStderr.write("\r[partial] \(transcript.text)\u{001B}[K")
                }
            case .commit(let transcript):
                if jsonl {
                    try write(.transcript(type: "commit", value: transcript))
                } else {
                    if !quiet { CLIStderr.write("\r\u{001B}[K") }
                    print(transcript.text)
                }
            case .stats(let stats, let queuedAudioMs):
                if jsonl {
                    try write(.stats(stats, queuedAudioMs: queuedAudioMs))
                }
            case .final(let reason):
                if jsonl {
                    try write(.final(reason))
                } else if !quiet {
                    CLIStderr.write("\r\u{001B}[K")
                }
            }
        }
    }

    static func errorCode(for error: Error) -> String {
        let message = error.localizedDescription
        if message.contains("backpressure_exceeded") { return "backpressure_exceeded" }
        if message.contains("pcm_odd_byte_eof") { return "pcm_odd_byte_eof" }
        if error is ASRStreamingError { return "invalid_stream" }
        return "runtime_error"
    }
}

enum CLIQwenASRLoader {
    static func prepare(
        model: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> Qwen3ASRGenerator {
        let normalized = model?.trimmingCharacters(in: .whitespacesAndNewlines)
        let value = normalized?.isEmpty == false ? normalized : nil
        let explicitPath = value.map { URL(fileURLWithPath: $0).standardizedFileURL }
            .flatMap { FileManager.default.fileExists(atPath: $0.path) ? $0 : nil }
        let modelId = explicitPath == nil ? (value ?? Qwen3ASRResources.defaultModelId) : Qwen3ASRResources.defaultModelId
        let generator = Qwen3ASRGenerator(modelId: modelId)
        if let explicitPath {
            try await generator.prepare(modelPath: explicitPath.path, progressHandler: progressHandler)
        } else if let localRoot = localModelRoot() {
            try await generator.prepare(modelPath: localRoot.path, progressHandler: progressHandler)
        } else {
            try await generator.prepare(progressHandler: progressHandler)
        }
        return generator
    }

    private static func localModelRoot() -> URL? {
        let fileManager = FileManager.default
        let base = MereRunModelPaths.resolveModelDir(Qwen3ASRResources.defaultModelId) { root in
            fileManager.fileExists(atPath: root.appendingPathComponent("config.json").path)
                || fileManager.fileExists(
                    atPath: root.appendingPathComponent(Qwen3ASRResources.defaultModelId)
                        .appendingPathComponent("config.json").path
                )
        }
        let nested = base.appendingPathComponent(Qwen3ASRResources.defaultModelId, isDirectory: true)
        if fileManager.fileExists(atPath: nested.appendingPathComponent("config.json").path) { return nested }
        if fileManager.fileExists(atPath: base.appendingPathComponent("config.json").path) { return base }
        return nil
    }
}

enum CLIParakeetASRLoader {
    static func prepare(
        model: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ParakeetGenerator {
        let normalized = model?.trimmingCharacters(in: .whitespacesAndNewlines)
        let value = normalized?.isEmpty == false ? normalized : nil
        let explicitPath = value.map { URL(fileURLWithPath: $0).standardizedFileURL }
            .flatMap { FileManager.default.fileExists(atPath: $0.path) ? $0 : nil }
        let modelId = explicitPath == nil ? (value ?? ParakeetResources.defaultModelId) : ParakeetResources.defaultModelId
        let generator = ParakeetGenerator(modelId: modelId)
        try await generator.prepare(
            modelPath: explicitPath?.path,
            progressHandler: progressHandler
        )
        return generator
    }
}
