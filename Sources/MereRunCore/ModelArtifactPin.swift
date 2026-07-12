import Crypto
import Foundation

public struct ModelArtifactPin: Codable, Equatable, Sendable {
    public let filename: String
    public let byteCount: Int64
    public let sha256: String

    public init(filename: String, byteCount: Int64, sha256: String) {
        self.filename = filename
        self.byteCount = byteCount
        self.sha256 = sha256.lowercased()
    }

    public func verify(in rootURL: URL, fileManager: FileManager = .default) throws -> URL {
        let url = rootURL.appendingPathComponent(filename)
        guard fileManager.fileExists(atPath: url.path) else {
            throw ModelArtifactVerificationError.missing(url.path)
        }
        // Managed installs are symlinks into the immutable Hub snapshot. Read
        // size and bytes from the resolved target while preserving the stable
        // managed path in diagnostics and return values.
        let verificationURL = url.resolvingSymlinksInPath()
        let actualByteCount = try Self.fileByteCount(url, fileManager: fileManager)
        guard actualByteCount == byteCount else {
            throw ModelArtifactVerificationError.sizeMismatch(
                path: url.path,
                expected: byteCount,
                actual: actualByteCount
            )
        }
        let actualSHA256 = try Self.fileSHA256(verificationURL)
        guard actualSHA256 == sha256 else {
            throw ModelArtifactVerificationError.checksumMismatch(
                path: url.path,
                expected: sha256,
                actual: actualSHA256
            )
        }
        return url
    }

    /// Returns the byte count of the file contents, following managed-install
    /// symlinks rather than reporting the length of the link itself.
    public static func fileByteCount(
        _ url: URL,
        fileManager: FileManager = .default
    ) throws -> Int64 {
        let verificationURL = url.standardizedFileURL.resolvingSymlinksInPath()
        let attributes = try fileManager.attributesOfItem(atPath: verificationURL.path)
        return (attributes[.size] as? NSNumber)?.int64Value ?? -1
    }

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
}

public enum ModelArtifactVerificationError: Error, Equatable, LocalizedError, Sendable {
    case missing(String)
    case sizeMismatch(path: String, expected: Int64, actual: Int64)
    case checksumMismatch(path: String, expected: String, actual: String)

    public var errorDescription: String? {
        switch self {
        case .missing(let path):
            "Pinned model artifact is missing: \(path)"
        case .sizeMismatch(let path, let expected, let actual):
            "Pinned model artifact has the wrong size at \(path): expected \(expected) bytes, found \(actual)."
        case .checksumMismatch(let path, let expected, let actual):
            "Pinned model artifact checksum mismatch at \(path): expected \(expected), found \(actual)."
        }
    }
}
