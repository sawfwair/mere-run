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

    /// Reading a closed protected record can require unlocking the device.
    /// Permission failures do not establish that a run was interrupted.
    public static func readData(at url: URL) throws -> Data {
        do {
            return try Data(contentsOf: url)
        } catch let error as CocoaError where error.code == .fileReadNoPermission {
            throw RunRecordReadIssue(url: url)
        } catch let error as POSIXError where error.code == .EACCES || error.code == .EPERM {
            throw RunRecordReadIssue(url: url)
        }
    }

    public static func write(_ value: some Encodable, to url: URL) throws {
        _ = try writeArtifact(value, to: url)
    }

    /// Returns provenance from the bytes committed through the open writer.
    /// A protected artifact need not be reopened after the device locks.
    public static func writeArtifact(_ value: some Encodable, to url: URL) throws -> RunArtifact {
        try RunFileWriter.write(encoder().encode(value), to: url)
    }
}

public struct RunRecordReadIssue: LocalizedError, Sendable {
    public let url: URL

    public var errorDescription: String? {
        "Cannot read run record: \(url.path). Unlock the device or check the file permissions, then retry."
    }
}
