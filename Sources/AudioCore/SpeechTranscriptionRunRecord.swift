import Foundation
import MereRunExecution

/// Supplied file-transcription settings before backend and model resolution.
public struct SpeechTranscriptionRunOptions: Codable, Equatable, Sendable {
    public let request: ASRRequest
    public let preferredBackend: ASRBackend
    public let modelOverride: String?
    public let provider: ParakeetExecutionProvider

    public init(
        request: ASRRequest, preferredBackend: ASRBackend, modelOverride: String? = nil,
        provider: ParakeetExecutionProvider = .mlx
    ) {
        self.request = request
        self.preferredBackend = preferredBackend
        self.modelOverride = modelOverride
        self.provider = provider
    }
}

/// Versioned history for one file transcription or translation. Live stdin
/// sessions retain their own streaming protocol and are not recorded here.
public struct SpeechTranscriptionRunRecord: Codable, Equatable, Sendable {
    public static let filename = "transcription-run.json"
    static let lockFilename = ".transcription-run.lock"

    public let schemaVersion: Int
    public let id: UUID
    public let parentID: UUID?
    public let createdAt: Date
    public var updatedAt: Date
    public var state: RunState
    public let requested: SpeechTranscriptionRunOptions
    public var effective: SpeechTranscriptionPlan?
    public var input: RunArtifact?
    public var artifacts: [RunArtifact]
    public var issue: SpeechTranscriptionIssue?

    public var canRetry: Bool {
        state.isTerminal && input != nil && effective?.modelPath != nil && effective?.modelMetadata.contains(where: { $0.artifact != nil }) == true
    }

    public static func recordURL(at url: URL) -> URL {
        url.lastPathComponent == filename ? url : url.appendingPathComponent(filename)
    }

    /// Recovers an abandoned run only while holding its process lease.
    public static func inspect(at url: URL) throws -> Self {
        let recordURL = recordURL(at: url)
        let lease = try RunDirectoryLease.acquire(in: recordURL.deletingLastPathComponent(), filename: lockFilename)
        defer { lease?.release() }
        var record = try decode(RunRecordCodec.readData(at: recordURL))
        if lease != nil, !record.state.isTerminal {
            record.state = .interrupted
            record.updatedAt = RunRecordCodec.timestamp()
            record.issue = SpeechTranscriptionIssue("process_interrupted", "The transcription process stopped before recording a terminal result.")
            try RunRecordCodec.write(record, to: recordURL)
        }
        return record
    }

    static func decode(_ data: Data) throws -> Self {
        let record = try RunRecordCodec.decoder().decode(Self.self, from: data)
        guard record.schemaVersion == 1 else {
            throw SpeechTranscriptionIssue("run_version_unsupported", "Unsupported transcription run record version: \(record.schemaVersion).")
        }
        return record
    }
}
