import CryptoKit
import Foundation

enum ArchiveIntegrity {
    enum IntegrityError: LocalizedError, Equatable {
        case invalidSHA256(String)
        case sha256Mismatch(expected: String, actual: String)

        var errorDescription: String? {
            switch self {
            case .invalidSHA256(let value):
                return "Invalid SHA-256 digest: \(value)"
            case .sha256Mismatch(let expected, let actual):
                return "Archive SHA-256 mismatch (got \(actual), expected \(expected))"
            }
        }
    }

    static func verify(file url: URL, expectedSHA256: String?) throws {
        guard let expected = try normalizedSHA256(expectedSHA256) else {
            return
        }
        let actual = try sha256Hex(of: url)
        guard actual == expected else {
            throw IntegrityError.sha256Mismatch(expected: expected, actual: actual)
        }
    }

    static func normalizedSHA256(_ value: String?) throws -> String? {
        guard let value else { return nil }
        let normalized = value
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard !normalized.isEmpty else { return nil }
        guard normalized.count == 64,
              normalized.allSatisfy({ $0.isHexDigit }) else {
            throw IntegrityError.invalidSHA256(value)
        }
        return normalized
    }

    static func sha256Hex(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 1024 * 1024)
            guard let chunk, !chunk.isEmpty else {
                break
            }
            hasher.update(data: chunk)
        }

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
