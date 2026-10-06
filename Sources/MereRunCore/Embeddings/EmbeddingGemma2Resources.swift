import Foundation

public enum EmbeddingGemma2Catalog {
    public static let modelID = "text-embed-embeddinggemma2"
    public static let repository = "google/embeddinggemma-2"
    public static let revision = "914f7f89142e33e77833254d9c9b90c3cef7303b"
    public static let hubFallback = HubFallbackConfig(
        repoId: repository, revision: revision,
        patterns: ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "processor_config.json", "preprocessor_config.json"]
    )
}

public struct EmbeddingGemma2Resources: Sendable {
    public let rootURL: URL
    public init(rootURL: URL) { self.rootURL = rootURL }
    public var configURL: URL { rootURL.appending(path: "config.json") }
    public var weightsURL: URL { rootURL.appending(path: "model.safetensors") }
    public var indexURL: URL { rootURL.appending(path: "model.safetensors.index.json") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var paths = [configURL, rootURL.appending(path: "tokenizer.json"), rootURL.appending(path: "tokenizer_config.json")]
        if !fileManager.fileExists(atPath: weightsURL.path), !fileManager.fileExists(atPath: indexURL.path) {
            paths.append(weightsURL)
        }
        return paths.filter { !fileManager.fileExists(atPath: $0.path) }
    }
}

public enum EmbeddingGemma2Task: String, CaseIterable, Sendable {
    case raw, query, document
    case codeRetrieval = "code-retrieval"
    case questionAnswering = "question-answering"
    case factChecking = "fact-checking"
    case classification, clustering
    case similarity

    public func format(_ text: String, title: String? = nil) -> String {
        switch self {
        case .raw: return text
        case .document: return "title: \(title ?? "none") | text: \(text)"
        case .query: return "task: search result | query: \(text)"
        case .codeRetrieval: return "task: code retrieval | query: \(text)"
        case .questionAnswering: return "task: question answering | query: \(text)"
        case .factChecking: return "task: fact checking | query: \(text)"
        case .classification: return "task: classification | query: \(text)"
        case .clustering: return "task: clustering | query: \(text)"
        case .similarity: return "task: sentence similarity | query: \(text)"
        }
    }
}
