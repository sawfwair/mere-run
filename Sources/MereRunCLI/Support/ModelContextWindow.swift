import Foundation

/// Reads declared context capacity without loading weights or invoking inference.
/// Missing or unfamiliar configurations intentionally report no capacity.
enum ModelContextWindow {
    private struct Configuration: Decodable {
        let maxPositionEmbeddings: Int?
        let contextLength: Int?
        let textConfig: TextConfiguration?
        enum CodingKeys: String, CodingKey {
            case maxPositionEmbeddings = "max_position_embeddings"
            case contextLength = "context_length"
            case textConfig = "text_config"
        }
    }
    private struct TextConfiguration: Decodable {
        let maxPositionEmbeddings: Int?
        enum CodingKeys: String, CodingKey { case maxPositionEmbeddings = "max_position_embeddings" }
    }

    static func read(at root: URL) -> Int? {
        guard let data = try? Data(contentsOf: root.appendingPathComponent("config.json")),
              let config = try? JSONDecoder().decode(Configuration.self, from: data),
              let tokens = config.textConfig?.maxPositionEmbeddings ?? config.maxPositionEmbeddings ?? config.contextLength,
              tokens > 0 else { return nil }
        return tokens
    }
}
