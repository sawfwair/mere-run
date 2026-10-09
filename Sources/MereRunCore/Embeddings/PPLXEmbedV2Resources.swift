import Foundation

public enum PPLXEmbedV2Catalog {
    public static let lateSmallID = "text-embed-pplx-v2-late-0.6b"
    public static let lateLargeID = "text-embed-pplx-v2-late-9b"
    public static let contextID = "text-embed-pplx-v2-context-9b-preview"
    public static let lateQuantizedID = "text-embed-pplx-v2-late-9b-mixed-4bit"
    public static let contextQuantizedID = "text-embed-pplx-v2-context-9b-preview-8bit"
    public static let modelIDs = [lateSmallID, lateLargeID, contextID, lateQuantizedID, contextQuantizedID]

    public static func isContextual(_ modelID: String) -> Bool {
        [contextID, contextQuantizedID].contains(modelID)
    }

    public static func estimatedDownloadBytes(_ modelID: String) -> Int64 {
        switch modelID {
        case lateSmallID: return 2_400_000_000
        case lateQuantizedID: return 7_400_000_000
        case contextQuantizedID: return 10_900_000_000
        default: return 33_700_000_000
        }
    }

    public static func repository(_ modelID: String) -> String {
        switch modelID {
        case lateQuantizedID: return "Sawfwair/pplx-embed-v2-late-9b-MLX-Mixed-4bit"
        case contextQuantizedID: return "Sawfwair/pplx-embed-v2-context-9b-preview-MLX-8bit"
        default: return "perplexity-ai/" + String(modelID.dropFirst("text-embed-".count)).replacingOccurrences(of: "pplx-v2", with: "pplx-embed-v2")
        }
    }

    public static func revision(_ modelID: String) -> String {
        switch modelID {
        case lateSmallID: return "dd4e95b836a73f6f0c32e46ea127c0b86b02169e"
        case lateQuantizedID: return "7309bd8d28fd8e0a9a27033f3d13cc2d7a5cf219"
        case contextQuantizedID: return "abf77a86a7b84aee72640c736a92f3c5b471e6db"
        case lateLargeID: return "77e936a1b18ed2ac00b7c76fccd70dc6a1bb1c18"
        default: return "b667039ee8b438a6350fbc91bbcecd86f9d363ba"
        }
    }

    public static func hubFallback(_ modelID: String) -> HubFallbackConfig {
        let patterns = [lateQuantizedID, contextQuantizedID].contains(modelID) ? ["*"]
            : ["config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json",
               "*.safetensors", "model.safetensors.index.json", "1_Dense/*", "2_MultiVectorMask/config.json", "README.md"]
        return HubFallbackConfig(repoId: repository(modelID), revision: revision(modelID), patterns: patterns)
    }
}

public struct PPLXEmbedV2Resources: Sendable {
    public let rootURL: URL
    public init(rootURL: URL) { self.rootURL = rootURL }
    public var configURL: URL { rootURL.appending(path: "config.json") }
    public var weightsURL: URL { rootURL.appending(path: "model.safetensors") }
    public var indexURL: URL { rootURL.appending(path: "model.safetensors.index.json") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var paths = [configURL, rootURL.appending(path: "tokenizer.json"), rootURL.appending(path: "tokenizer_config.json")]
        if !fileManager.fileExists(atPath: weightsURL.path), !fileManager.fileExists(atPath: indexURL.path) { paths.append(weightsURL) }
        if let config = try? JSONDecoder().decode(PPLXEmbedV2Config.self, from: Data(contentsOf: configURL)), !config.isContextual {
            paths += [rootURL.appending(path: "1_Dense/config.json"), rootURL.appending(path: "1_Dense/model.safetensors"),
                      rootURL.appending(path: "2_MultiVectorMask/config.json")]
        }
        return paths.filter { !fileManager.fileExists(atPath: $0.path) }
    }
}

public enum PPLXEmbedV2Error: LocalizedError {
    case invalidConfiguration(String)
    case invalidInput(String)
    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message), .invalidInput(let message): return message
        }
    }
}

public enum PPLXEmbedV2Task: String, Sendable { case query, document }

/// A row contains token vectors (late) or chunk vectors (contextual).
public struct PPLXEmbedV2Result: Codable, Sendable {
    public let model: String
    public let representation: String
    public let dimensions: Int
    public let normalized: Bool
    public let data: [Row]
    public struct Row: Codable, Sendable {
        public let index: Int
        public let embeddings: [[Float]]
        public let tokenCount: Int
    }

    public static func maxSim(query: [[Float]], document: [[Float]]) throws -> Float {
        guard let dimensions = query.first?.count, dimensions > 0, !document.isEmpty,
              (query + document).allSatisfy({ $0.count == dimensions && $0.allSatisfy(\.isFinite) }) else {
            throw PPLXEmbedV2Error.invalidInput("MaxSim requires nonempty, finite token vectors with matching dimensions.")
        }
        return query.reduce(Float(0)) { score, vector in
            score + document.map { zip(vector, $0).reduce(Float(0)) { $0 + $1.0 * $1.1 } }.max()!
        }
    }
}
