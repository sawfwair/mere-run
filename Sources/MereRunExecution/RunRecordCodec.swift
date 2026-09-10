import Crypto
import Foundation

/// Shared encoding and atomic file writes. Each operation owns its record schema
/// and validates supported versions before recovering or retrying a run.
public enum RunRecordCodec {
    public static func timestamp() -> Date {
        Date(timeIntervalSince1970: Date().timeIntervalSince1970.rounded(.down))
    }

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

    public static func digest(_ value: some Encodable) throws -> String {
        SHA256.hash(data: try encoder().encode(value)).map { String(format: "%02x", $0) }.joined()
    }

    public static func write(_ value: some Encodable, to url: URL) throws {
        try encoder().encode(value).write(to: url, options: [.atomic, .completeFileProtectionUnlessOpen])
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }
}
