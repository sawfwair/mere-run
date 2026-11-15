import Foundation

/// Portable native weights. Unlike the local conversion cache, records have no file timestamps.
public struct SOVTranscribeBundleManifest: Codable, Sendable {
    public static let filename = "sov-bundle.json"
    public let formatVersion: Int
    public let packerVersion: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let configSHA256: String
    public let groupSize: Int
    public let layerBits: [Int]
    public let audio: FileRecord
    public let shards: [FileRecord]

    public struct FileRecord: Codable, Sendable {
        public let name: String
        public let bytes: Int
        public let headerSHA256: String
        public let sha256: String
    }

    public enum ValidationError: LocalizedError {
        case invalid
        public var errorDescription: String? { "Invalid or unsupported portable SOV bundle manifest." }
    }

    public func validate() throws {
        let files = [audio] + shards
        func hash(_ value: String) -> Bool {
            value.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
        }
        guard formatVersion == 1, packerVersion == "affine-fused-gate-up-v1",
              !sourceRepository.isEmpty, !sourceRevision.isEmpty, hash(configSHA256),
              groupSize == 64, !layerBits.isEmpty, layerBits.allSatisfy({ $0 == 8 }),
              audio.name == "model.safetensors", !shards.isEmpty,
              Set(files.map(\.name)).count == files.count,
              files.allSatisfy({
                  !$0.name.isEmpty && URL(fileURLWithPath: $0.name).lastPathComponent == $0.name
                      && $0.name.hasSuffix(".safetensors") && $0.bytes > 0 && hash($0.headerSHA256) && hash($0.sha256)
              }) else { throw ValidationError.invalid }
    }
}
