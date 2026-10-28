import Crypto
import Foundation

/// A file's location, content digest, and size at the time it was recorded.
public struct RunArtifact: Codable, Hashable, Sendable {
    public let url: URL
    public let sha256: String
    public let byteCount: UInt64

    public init(url: URL, sha256: String, byteCount: UInt64) {
        self.url = url
        self.sha256 = sha256
        self.byteCount = byteCount
    }

    public static func read(_ url: URL) throws -> Self {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        var byteCount: UInt64 = 0
        // Foundation's autoreleased read buffers otherwise accumulate until
        // the caller's pool drains when hashing multi-gigabyte checkpoints.
        func readChunk() throws -> Data? {
            #if canImport(ObjectiveC)
            return try autoreleasepool { try handle.read(upToCount: 1_048_576) }
            #else
            return try handle.read(upToCount: 1_048_576)
            #endif
        }
        while let bytes = try readChunk(), !bytes.isEmpty {
            hash.update(data: bytes)
            byteCount += UInt64(bytes.count)
        }
        return Self(url: url, sha256: hash.finalize().map { String(format: "%02x", $0) }.joined(), byteCount: byteCount)
    }
}
