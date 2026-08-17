import Crypto
import Foundation

/// Canonical JSON encoding for every workflow bundle, run, and relay wire
/// document: pretty sorted keys for files, compact sorted keys for NDJSON
/// lines and fingerprint hashing, ISO-8601 dates throughout.
public enum WorkflowBundleCodec {
    public static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    public static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }

    public static func lineEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    public static func write<T: Encodable>(_ value: T, to url: URL) throws {
        try encoder().encode(value).write(to: url, options: .atomic)
    }

    public static func hash<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return SHA256.hash(data: try encoder.encode(value)).map { String(format: "%02x", $0) }.joined()
    }
}

/// SHA-256 helpers shared by relay asset upload and artifact verification.
public enum ModelArtifactPinDigest {
    public static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Streaming file digest, byte-for-byte equivalent to the model-store
    /// implementation in MereRunCore so fetch verification matches installs.
    public static func fileSHA256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 1_048_576) ?? Data()
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    public static func fileByteCount(
        _ url: URL,
        fileManager: FileManager = .default
    ) throws -> Int64 {
        let verificationURL = url.standardizedFileURL.resolvingSymlinksInPath()
        let attributes = try fileManager.attributesOfItem(atPath: verificationURL.path)
        return (attributes[.size] as? NSNumber)?.int64Value ?? -1
    }
}
