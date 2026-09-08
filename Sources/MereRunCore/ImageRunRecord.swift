import Crypto
import Foundation

/// Versioned image history. Requested settings retain omissions; resolved settings
/// contain the values submitted to the runtime, including a seed chosen before execution.
public struct ImageRunRecord: Codable, Equatable, Sendable {
    public static let filename = "image-run.json"
    public enum State: String, Codable, Sendable {
        case preparing, running, succeeded, failed, cancelled, interrupted
        public var isTerminal: Bool { self != .preparing && self != .running }
    }

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

    public struct Artifact: Codable, Equatable, Sendable {
        public let url: URL
        public let sha256: String
        public let byteCount: UInt64

        public static func read(_ url: URL) throws -> Self {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            var hash = SHA256()
            var byteCount: UInt64 = 0
            while let bytes = try handle.read(upToCount: 1_048_576), !bytes.isEmpty {
                hash.update(data: bytes)
                byteCount += UInt64(bytes.count)
            }
            return Self(url: url, sha256: hash.finalize().map { String(format: "%02x", $0) }.joined(), byteCount: byteCount)
        }
    }

    public static func recordURL(at url: URL) -> URL {
        url.lastPathComponent == filename ? url : url.appendingPathComponent(filename)
    }

    /// A lock held by the executing process is the source of liveness. Reading an
    /// abandoned nonterminal record persists an interrupted state, even after reboot.
    public static func inspect(at url: URL) throws -> Self {
        let recordURL = recordURL(at: url)
        let lease = try ImageRunLease.acquire(in: recordURL.deletingLastPathComponent())
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
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let record = try decoder.decode(Self.self, from: data)
        guard record.schemaVersion == 1 else {
            throw ImageGenerationIssue("run_version_unsupported", "Unsupported image run record version: \(record.schemaVersion).")
        }
        return record
    }

    static func timestamp() -> Date {
        Date(timeIntervalSince1970: Date().timeIntervalSince1970.rounded(.down))
    }

    static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    static func digest(_ manifest: MereRunModelManifest) throws -> String {
        SHA256.hash(data: try encoder().encode(manifest)).map { String(format: "%02x", $0) }.joined()
    }

    func write(to url: URL) throws {
        try Self.encoder().encode(self).write(to: url, options: [.atomic, .completeFileProtectionUnlessOpen])
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }
}
