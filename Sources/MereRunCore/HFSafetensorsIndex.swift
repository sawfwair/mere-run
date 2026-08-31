import Foundation

/// Hugging Face sharded safetensors index format (`*.safetensors.index.json`).
public struct HFSafetensorsIndex: Codable, Sendable, Hashable {
    public struct Metadata: Codable, Sendable, Hashable {
        public let totalSize: Int64?

        enum CodingKeys: String, CodingKey {
            case totalSize = "total_size"
        }
    }

    public let metadata: Metadata?
    public let weightMap: [String: String]

    public init(metadata: Metadata?, weightMap: [String: String]) {
        self.metadata = metadata
        self.weightMap = weightMap
    }

    enum CodingKeys: String, CodingKey {
        case metadata
        case weightMap = "weight_map"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.metadata = try container.decodeIfPresent(Metadata.self, forKey: .metadata)
        self.weightMap = try container.decode([String: String].self, forKey: .weightMap)
    }

    public var shardFilenames: [String] {
        Array(Set(weightMap.values)).sorted()
    }
}
