import Foundation

public enum LightOnOCRMode: String, Sendable {
    case plain
    case grounding

    public var prompt: String { self == .grounding ? "grounding" : "" }
}

public enum LightOnOCRResources {
    public static let oneBRevision = "b9a2b4c17f1eee9f29058d716b66b5f8e7d8db86"

    struct Architecture: Decodable {
        enum ModelType: String, Decodable { case mistral3, qwen35 = "qwen3_5" }
        let modelType: ModelType
        enum CodingKeys: String, CodingKey { case modelType = "model_type" }
    }
}
