import MereRunExecution
import Foundation

/// Versioned image history. Requested settings retain omissions; resolved settings
/// contain the values submitted to the runtime, including a seed chosen before execution.
public struct ImageRunRecord: Codable, Equatable, Sendable {
    public static let filename = "image-run.json"
    public typealias State = RunState
    public typealias Artifact = RunArtifact

    public let schemaVersion: Int
    public let id: UUID
    public let parentID: UUID?
    public let createdAt: Date
    public var updatedAt: Date
    public var state: State
    public let modelSelector: String
    public let requested: ImageGenerationOptions
    public var resolvedOptions: ImageGenerationOptions?
    public var effective: GenerationRequest?
    public var policy: ImageGenerationPolicy?
    public var modelRoot: URL?
    public var modelManifest: MereRunModelManifest?
    public var manifestDigest: String?
    public var installedManifestDigest: String?
    public var backend: ImageGenerationBackend?
    public var inputs: [Artifact]
    public var artifacts: [Artifact]
    public var issue: ImageGenerationIssue?

    public static func recordURL(at url: URL) -> URL {
        url.lastPathComponent == filename ? url : url.appendingPathComponent(filename)
    }

    /// A lock held by the executing process is the source of liveness. Reading an
    /// abandoned nonterminal record persists an interrupted state, even after reboot.
    public static func inspect(at url: URL) throws -> Self {
        let recordURL = recordURL(at: url)
        let lease = try RunDirectoryLease.acquire(in: recordURL.deletingLastPathComponent(), filename: ".image-run.lock")
        defer { lease?.release() }
        var record = try decode(Data(contentsOf: recordURL))
        if lease != nil, !record.state.isTerminal {
            record.state = .interrupted
            record.updatedAt = timestamp()
            record.issue = ImageGenerationIssue("process_interrupted", "The image process stopped before recording a terminal result.")
            try record.write(to: recordURL)
        }
        return record
    }

    static func decode(_ data: Data) throws -> Self {
        let record = try RunRecordCodec.decoder().decode(Self.self, from: data)
        guard record.schemaVersion == 1 else {
            throw ImageGenerationIssue("run_version_unsupported", "Unsupported image run record version: \(record.schemaVersion).")
        }
        return record
    }

    static func timestamp() -> Date {
        RunRecordCodec.timestamp()
    }

    static func encoder() -> JSONEncoder {
        RunRecordCodec.encoder()
    }

    static func digest(_ manifest: MereRunModelManifest) throws -> String {
        try RunRecordCodec.digest(manifest)
    }

    func write(to url: URL) throws {
        try RunRecordCodec.write(self, to: url)
    }
}
