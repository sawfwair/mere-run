import Foundation
import MereRunCoreAIModel

public enum EmbeddingGemma2CoreAICatalog {
    public static let modelID = "text-embed-embeddinggemma2-coreai"
    public static let repository = "EverPaper/embeddinggemma-2-coreai"
    public static let revision = "305fc2a96790d9343f38038aeca211dfca56d9a5"
    public static let hubFallback = HubFallbackConfig(
        repoId: repository, revision: revision,
        patterns: ["model/*", "LICENSE", "NOTICE", "CHECKSUMS.sha256"]
    )
    public static let assets = ["embeddinggemma-2-text-mm.aimodel", "embeddinggemma-2-vision.aimodel", "embeddinggemma-2-audio.aimodel"]

    /// Accept the repository snapshot root or its model/ directory for local paths.
    public static func bundleRoot(_ root: URL, fileManager: FileManager = .default) -> URL {
        if fileManager.fileExists(atPath: root.appending(path: "config.json").path) { return root }
        let nested = root.appending(path: "model")
        return fileManager.fileExists(atPath: nested.appending(path: "config.json").path) ? nested : root
    }

    public static func missingFiles(root: URL, fileManager: FileManager = .default) -> [URL] {
        let bundle = bundleRoot(root, fileManager: fileManager)
        let paths = ["config.json", "MODEL-ATTRIBUTION.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
            + assets.flatMap { asset in ["main.hash", "main.mlirb", "metadata.json"].map { asset + "/" + $0 } }
        return paths.map { bundle.appending(path: $0) }.filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public static func validateConfiguration(root: URL) throws {
        let bundle = bundleRoot(root)
        let config = try JSONDecoder().decode(BundleConfig.self, from: Data(contentsOf: bundle.appending(path: "config.json")))
        try config.validate()
    }
}

public enum EmbeddingGemma2CoreAIRuntime {
    public static var isAvailable: Bool {
        #if canImport(CoreAI) && os(macOS) && arch(arm64)
        if #available(macOS 27.0, *) { return true }
        #endif
        return false
    }

    public static func requireAvailable() throws {
        guard isAvailable else {
            throw EmbeddingGemma2Error.invalidConfiguration(
                "EmbeddingGemma 2 Core AI requires Apple Silicon, macOS 27, and a mere.run build made with the macOS 27 SDK."
            )
        }
    }

    public static func validateInputs(_ inputs: [EmbeddingGemma2Input], task: EmbeddingGemma2Task,
                                      title: String?, dimensions: Int, maxTokens: Int?) throws {
        guard !inputs.isEmpty, inputs.allSatisfy({ !$0.content.isEmpty }),
              [128, 256, 512, 768].contains(dimensions), title == nil || task == .document,
              maxTokens == nil || maxTokens! >= 2 else {
            throw EmbeddingGemma2Error.invalidInput("Provide nonempty inputs, 128/256/512/768 dimensions, and at least two tokens. A title requires --task document.")
        }
        for part in inputs.flatMap(\.content) {
            switch part {
            case .video, .videoFrames:
                throw EmbeddingGemma2Error.invalidInput("The pinned Core AI export supports text, images, and audio. Use text-embed-embeddinggemma2 for video.")
            case .image(let url), .audio(let url):
                guard url.isFileURL else { throw EmbeddingGemma2Error.invalidInput("Media inputs require local file URLs.") }
            case .text: break
            }
        }
    }

    public static func embed(root: URL, inputs: [EmbeddingGemma2Input], task: EmbeddingGemma2Task = .raw,
                             title: String? = nil, dimensions: Int = 768, maxTokens: Int? = nil) async throws
        -> (embeddings: [[Float]], tokenCounts: [Int]) {
        try validateInputs(inputs, task: task, title: title, dimensions: dimensions, maxTokens: maxTokens)
        try requireAvailable()
        let missing = EmbeddingGemma2CoreAICatalog.missingFiles(root: root)
        guard missing.isEmpty else {
            throw EmbeddingGemma2Error.invalidConfiguration("Missing Core AI resources:\n" + missing.map(\.path).joined(separator: "\n"))
        }
        try EmbeddingGemma2CoreAICatalog.validateConfiguration(root: root)
        #if canImport(CoreAI) && os(macOS) && arch(arm64)
        if #available(macOS 27.0, *) {
            return try await execute(root: EmbeddingGemma2CoreAICatalog.bundleRoot(root), inputs: inputs,
                                     task: task, title: title, dimensions: dimensions, maxTokens: maxTokens)
        }
        #endif
        throw EmbeddingGemma2Error.invalidConfiguration("Core AI is unavailable in this build.")
    }
}
