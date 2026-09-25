import Foundation
import MLX
import MereRunGLiNERModel
@preconcurrency import Hub
@preconcurrency import Tokenizers

public struct GLiNERClassificationTask: Codable, Sendable {
    public let name: String
    public let labels: [String]
    public let descriptions: [String: String]?
    public let prompt: String?
    public let multiLabel: Bool
    public let threshold: Double

    enum CodingKeys: String, CodingKey {
        case name, labels, descriptions, prompt
        case multiLabel = "multi_label", threshold
    }

    public init(name: String, labels: [String], descriptions: [String: String]? = nil,
                prompt: String? = nil, multiLabel: Bool = false, threshold: Double = 0.5) {
        self.name = name
        self.labels = labels
        self.descriptions = descriptions
        self.prompt = prompt
        self.multiLabel = multiLabel
        self.threshold = threshold
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        labels = try container.decode([String].self, forKey: .labels)
        descriptions = try container.decodeIfPresent([String: String].self, forKey: .descriptions)
        prompt = try container.decodeIfPresent(String.self, forKey: .prompt)
        multiLabel = try container.decodeIfPresent(Bool.self, forKey: .multiLabel) ?? false
        threshold = try container.decodeIfPresent(Double.self, forKey: .threshold) ?? 0.5
    }

    func validate() throws {
        let terms = [name] + labels + [prompt].compactMap { $0 } + (descriptions?.values.map { $0 } ?? [])
        guard !name.isEmpty, !labels.isEmpty, labels.count <= 64,
              Set(labels).count == labels.count, (0...1).contains(threshold),
              terms.allSatisfy({ !$0.isEmpty && $0.count <= 512 && !Self.hasMarker($0) }),
              descriptions?.keys.allSatisfy({ labels.contains($0) }) ?? true else {
            throw GLiNERClassificationError.invalidRequest("Task \(name) has invalid labels, threshold, or reserved markers.")
        }
    }

    private static func hasMarker(_ value: String) -> Bool {
        ["[P]", "[L]", "[C]", "[E]", "[R]", "[SEP_STRUCT]", "[SEP_TEXT]", "[DESCRIPTION]",
         "[EXAMPLE]", "[OUTPUT]"].contains { value.contains($0) }
    }
}

public struct GLiNERClassificationRequest: Codable, Sendable {
    public let text: String
    public let tasks: [GLiNERClassificationTask]

    public init(text: String, tasks: [GLiNERClassificationTask]) {
        self.text = text
        self.tasks = tasks
    }

    public func validate() throws {
        guard !text.isEmpty, !tasks.isEmpty, tasks.count <= 16,
              Set(tasks.map(\.name)).count == tasks.count else {
            throw GLiNERClassificationError.invalidRequest("Provide text and 1–16 unique classification tasks.")
        }
        try tasks.forEach { try $0.validate() }
    }
}

public struct GLiNERClassificationPlan: Codable, Sendable {
    public let model: String
    public let inputTokens: Int
    public let labelCount: Int
    public let taskNames: [String]
}

public struct GLiNERClassificationHead: Codable, Sendable {
    public let labels: [String]
    public let probabilities: [String: Double]
}

public struct GLiNERClassificationResponse: Codable, Sendable {
    public let model: String
    public let runtime: String
    public let heads: [String: GLiNERClassificationHead]
    public let inputTokens: Int
}

public enum GLiNERClassificationError: LocalizedError {
    case invalidRequest(String)

    public var errorDescription: String? {
        switch self {
        case .invalidRequest(let detail): detail
        }
    }
}

struct GLiNERSpecialTokenConfig: Decodable {
    struct Entry: Decodable {
        let content: String
    }

    let addedTokensDecoder: [String: Entry]

    enum CodingKeys: String, CodingKey {
        case addedTokensDecoder = "added_tokens_decoder"
    }

    var tokenIDs: [String: Int] {
        Dictionary(uniqueKeysWithValues: addedTokensDecoder.compactMap { key, value in
            Int(key).map { (value.content, $0) }
        })
    }
}

public final class GLiNERClassificationOperation {
    public let modelID: String
    private let root: URL
    private let configuration: GLiNEREncoderConfiguration
    private let tokenizer: any Tokenizer
    private let specialTokenIDs: [String: Int]
    private var network: GLiNERNetwork?

    public init(root: URL, modelID: String) throws {
        self.root = root
        self.modelID = modelID
        configuration = try JSONDecoder().decode(GLiNEREncoderConfiguration.self,
            from: Data(contentsOf: root.appending(path: "encoder_config/config.json")))
        try configuration.validate()
        let tokenizerConfig = try HubApi.shared.configuration(fileURL: root.appending(path: "tokenizer_config.json"))
        let tokenizerData = try HubApi.shared.configuration(fileURL: root.appending(path: "tokenizer.json"))
        let tokenConfig = try JSONDecoder().decode(GLiNERSpecialTokenConfig.self,
            from: Data(contentsOf: root.appending(path: "tokenizer_config.json")))
        specialTokenIDs = tokenConfig.tokenIDs
        guard specialTokenIDs["[P]"] == 128_003, specialTokenIDs["[L]"] == 128_007,
              specialTokenIDs["[SEP_STRUCT]"] == 128_001,
              specialTokenIDs["[SEP_TEXT]"] == 128_002 else {
            throw GLiNERClassificationError.invalidRequest("Tokenizer marker IDs do not match the checkpoint.")
        }
        // Swift Transformers does not register DebertaV2Tokenizer, although the
        // checkpoint's tokenizer.json declares a Unigram model. Select its
        // registered Unigram implementation without changing the snapshot.
        var unigramConfig = tokenizerConfig.dictionary(or: [:])
        unigramConfig["tokenizer_class"] = Config("XLMRobertaTokenizer")
        tokenizer = try AutoTokenizer.from(tokenizerConfig: Config(unigramConfig), tokenizerData: tokenizerData)
    }

    public func prepare(_ request: GLiNERClassificationRequest) throws -> GLiNERClassificationPlan {
        let prepared = try encode(request)
        return GLiNERClassificationPlan(model: modelID, inputTokens: prepared.ids.count,
                                        labelCount: prepared.markers.count, taskNames: request.tasks.map(\.name))
    }

    public func predict(_ request: GLiNERClassificationRequest) throws -> GLiNERClassificationResponse {
        let prepared = try encode(request)
        if network == nil {
            network = try GLiNERNetwork(configuration: configuration,
                arrays: MLX.loadArrays(url: root.appending(path: "model.safetensors")))
        }
        guard let network else { throw GLiNERClassificationError.invalidRequest("Model did not load.") }
        let logits = network(inputIDs: MLXArray(prepared.ids).reshaped([1, prepared.ids.count]),
                             markers: MLXArray(prepared.markers).reshaped([1, prepared.markers.count]))
        eval(logits)
        let scores = logits[0].asArray(Float.self).map(Double.init)
        var heads: [String: GLiNERClassificationHead] = [:]
        var offset = 0
        for task in request.tasks {
            let slice = Array(scores[offset..<(offset + task.labels.count)])
            let probabilities: [Double]
            if task.multiLabel {
                probabilities = slice.map { 1 / (1 + exp(-$0)) }
            } else {
                let peak = slice.max() ?? 0
                let values = slice.map { exp($0 - peak) }
                let total = values.reduce(0, +)
                probabilities = values.map { $0 / total }
            }
            let best = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
            var selected = task.multiLabel ? task.labels.indices.filter { probabilities[$0] >= task.threshold } : [best]
            if selected.isEmpty { selected = [best] }
            heads[task.name] = GLiNERClassificationHead(
                labels: selected.map { task.labels[$0] },
                probabilities: Dictionary(uniqueKeysWithValues: zip(task.labels, probabilities)))
            offset += task.labels.count
        }
        return GLiNERClassificationResponse(model: modelID, runtime: "native-swift-mlx-fp32",
                                            heads: heads, inputTokens: prepared.ids.count)
    }

    func encode(_ request: GLiNERClassificationRequest) throws -> (ids: [Int], markers: [Int]) {
        try request.validate()
        var parts: [String] = []
        for task in request.tasks {
            var prompt = task.name
            if let instruction = task.prompt { prompt += ": \(instruction)" }
            if let descriptions = task.descriptions {
                for label in task.labels {
                    if let description = descriptions[label] {
                        prompt += " [DESCRIPTION] \(label): \(description)"
                    }
                }
            }
            if !parts.isEmpty { parts.append("[SEP_STRUCT]") }
            parts += ["(", "[P]", prompt, "("]
            for label in task.labels { parts += ["[L]", label] }
            parts += [")", ")"]
        }
        parts.append("[SEP_TEXT]")
        let expression = try NSRegularExpression(pattern:
            #"(?:https?://[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|\w+(?:[-_]\w+)*|\S"#,
            options: [.caseInsensitive])
        let text = request.text as NSString
        let words = expression.matches(in: request.text, range: NSRange(location: 0, length: text.length))
            .map { text.substring(with: $0.range).lowercased() }
        parts += words
        var ids: [Int] = []
        var markers: [Int] = []
        for part in parts {
            if part == "[L]" { markers.append(ids.count) }
            ids += tokenizer.tokenize(text: part).map {
                specialTokenIDs[$0] ?? tokenizer.convertTokenToId($0) ?? 3
            }
        }
        guard !ids.isEmpty, ids.count <= configuration.maxPositionEmbeddings,
              ids.allSatisfy({ (0..<configuration.vocabSize).contains($0) }),
              markers.count == request.tasks.reduce(0, { $0 + $1.labels.count }) else {
            throw GLiNERClassificationError.invalidRequest("Schema and text exceed the 512 token checkpoint limit.")
        }
        return (ids, markers)
    }
}
